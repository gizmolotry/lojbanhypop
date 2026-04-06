from __future__ import annotations

import argparse
import json
import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from lojban_evolution.experiment import generate_dataset, split_dataset


@dataclass
class TrainItem:
    prompt: str
    answer: str


@contextmanager
def adapter_disabled(model):
    disable_ctx = None
    if hasattr(model, "disable_adapter"):
        disable_ctx = model.disable_adapter()
    elif hasattr(model, "disable_adapters"):
        disable_ctx = model.disable_adapters()
    if disable_ctx is None:
        with nullcontext():
            yield
    else:
        with disable_ctx:
            yield


def _resolve_layers(model):
    for root in (model, getattr(model, "model", None), getattr(model, "base_model", None)):
        if root is None:
            continue
        if hasattr(root, "layers"):
            return root.layers
        inner = getattr(root, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return inner.layers
    return None


@contextmanager
def persistent_inject(model, layer_index: int, inject_state: torch.Tensor, scale: float):
    layers = _resolve_layers(model)
    if layers is None:
        raise RuntimeError("Unable to locate decoder layers.")
    if not (0 <= layer_index < len(layers)):
        raise RuntimeError(f"layer_index out of range: {layer_index}")

    def _hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        add = inject_state.to(device=hidden.device, dtype=hidden.dtype)
        if add.dim() == 2:
            add = add.unsqueeze(1)
        hidden = hidden + (float(scale) * add)
        if rest is None:
            return hidden
        return (hidden, *rest)

    h = layers[layer_index].register_forward_hook(_hook)
    try:
        yield
    finally:
        h.remove()


class VQCodebook(torch.nn.Module):
    def __init__(self, codebook_size: int, hidden_size: int):
        super().__init__()
        self.emb = torch.nn.Parameter(torch.empty(codebook_size, hidden_size))
        torch.nn.init.normal_(self.emb, mean=0.0, std=0.02)

    def forward(self, z: torch.Tensor):
        # z: [B, 1, H]
        b, t, h = z.shape
        flat = z.reshape(b * t, h)  # [N,H]
        # nearest neighbor quantization
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ self.emb.t()
            + self.emb.pow(2).sum(dim=1, keepdim=True).t()
        )  # [N,K]
        idx = torch.argmin(dist, dim=1)  # [N]
        z_q = self.emb[idx].view(b, t, h)
        # straight-through estimator
        z_st = z + (z_q - z).detach()
        # VQ losses
        codebook_loss = F.mse_loss(z_q, z.detach())
        commit_loss = F.mse_loss(z, z_q.detach())
        return z_st, idx.view(b, t), codebook_loss, commit_loss


def build_logic_prompt(q: str) -> str:
    return (
        "You are a rigid symbolic reasoner.\n"
        "Output a concise symbolic TRACE and then ANSWER.\n\n"
        f"QUESTION: {q}\n"
        "TRACE:"
    )


def build_final_prefix(q: str) -> str:
    return (
        "Solve the logic question. Return only the final answer with no explanation.\n\n"
        f"Question: {q}\n"
        "Final answer:"
    )


def extract_last_hidden(model, tokenizer, question: str, max_logic_new_tokens: int) -> torch.Tensor:
    logic_prompt = build_logic_prompt(question)
    ids = tokenizer(logic_prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=True,
            return_dict=True,
            output_hidden_states=True,
        )
    past = out.past_key_values
    tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    h = out.hidden_states[-1][:, -1:, :]
    cur_len = ids.shape[1] + 1
    if tokenizer.eos_token_id is not None and int(tok.item()) == int(tokenizer.eos_token_id):
        return h
    for _ in range(max_logic_new_tokens - 1):
        am = torch.ones((1, cur_len + 1), dtype=torch.long, device=model.device)
        with torch.no_grad():
            out = model(
                input_ids=tok,
                attention_mask=am,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
        past = out.past_key_values
        tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        h = out.hidden_states[-1][:, -1:, :]
        cur_len += 1
        if tokenizer.eos_token_id is not None and int(tok.item()) == int(tokenizer.eos_token_id):
            break
    return h


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VQ reasoning pilot: learn emergent discrete codebook at layer-12.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True, help="Reasoning extractor adapter (system-1).")
    p.add_argument("--dataset-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--sample-size", type=int, default=128)
    p.add_argument("--train-steps", type=int, default=500)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--codebook-size", type=int, default=200)
    p.add_argument("--layer-index", type=int, default=12)
    p.add_argument("--inject-scale", type=float, default=1.0)
    p.add_argument("--commitment-weight", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--output-codebook", type=Path, default=Path("runs/projections/vq_codebook_pilot.pt"))
    p.add_argument("--output-report", type=Path, default=Path("runs/vq_reasoning_pilot_report.json"))
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    adapter_has_tokenizer = (args.adapter / "tokenizer.json").exists() or (args.adapter / "tokenizer_config.json").exists()
    tokenizer_source = str(args.adapter) if adapter_has_tokenizer else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, str(args.adapter), local_files_only=args.local_files_only)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    dataset = generate_dataset(size=args.dataset_size, seed=args.seed)
    _, _, test = split_dataset(dataset)
    sample = [TrainItem(prompt=x.prompt, answer=x.answer) for x in test[: args.sample_size]]
    if not sample:
        raise RuntimeError("Empty sample.")

    codebook = VQCodebook(codebook_size=args.codebook_size, hidden_size=int(model.config.hidden_size)).to(model.device, dtype=model.dtype)
    codebook.train()
    opt = torch.optim.AdamW(codebook.parameters(), lr=args.lr)

    emb = model.get_input_embeddings()
    losses: List[float] = []
    usage = torch.zeros(args.codebook_size, dtype=torch.long)

    for step in range(args.train_steps):
        it = sample[step % len(sample)]
        with torch.no_grad():
            h_t = extract_last_hidden(model, tokenizer, it.prompt, max_logic_new_tokens=args.max_logic_new_tokens).to(model.dtype)
        z_st, idx, cb_loss, commit_loss = codebook(h_t)
        usage[int(idx[0, 0].item())] += 1

        prefix = build_final_prefix(it.prompt)
        prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(model.device)
        target_ids = tokenizer(" " + it.answer, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        if int(target_ids.numel()) == 0:
            continue
        tgt = target_ids[:, 0]

        opt.zero_grad(set_to_none=True)
        with adapter_disabled(model):
            prefix_emb = emb(prefix_ids)
            am = torch.ones((1, prefix_emb.shape[1]), dtype=torch.long, device=model.device)
            with persistent_inject(model, layer_index=args.layer_index, inject_state=z_st, scale=args.inject_scale):
                out = model(inputs_embeds=prefix_emb, attention_mask=am, return_dict=True, use_cache=False)
        ce = F.cross_entropy(out.logits[:, -1, :], tgt)
        loss = ce + cb_loss + (float(args.commitment_weight) * commit_loss)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    args.output_codebook.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": codebook.state_dict(), "codebook_size": args.codebook_size}, args.output_codebook)

    used = int((usage > 0).sum().item())
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "codebook_size": args.codebook_size,
        "codes_used": used,
        "usage_ratio": float(used / max(1, args.codebook_size)),
        "train_steps": args.train_steps,
        "loss_start": float(losses[0]) if losses else 0.0,
        "loss_end": float(losses[-1]) if losses else 0.0,
        "loss_history_tail": losses[-20:],
        "output_codebook": str(args.output_codebook),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote: {args.output_codebook}")
    print(f"Wrote: {args.output_report}")
    print(f"codes_used: {used}/{args.codebook_size}")


if __name__ == "__main__":
    main()
