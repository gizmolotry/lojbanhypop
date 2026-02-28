from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from lojban_evolution.experiment import generate_dataset, split_dataset


NON_ALNUM_RE = re.compile(r"[^a-z0-9=,]+")
ROLE_RE = re.compile(
    r"\b([abc])\b[^a-zA-Z]{0,6}(?:is|:)?[^a-zA-Z]{0,6}(?:a|an)?[^a-zA-Z]{0,6}(knight|knave)\b",
    re.IGNORECASE,
)


@dataclass
class TrainItem:
    prompt: str
    target_answer: str


class SwiGLUBridge(torch.nn.Module):
    def __init__(self, hidden_size: int, expansion_factor: int = 2):
        super().__init__()
        inner = int(hidden_size * expansion_factor)
        self.w1 = torch.nn.Linear(hidden_size, inner, bias=True)
        self.w2 = torch.nn.Linear(inner, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU(x) = (xW1 ⊙ SiLU(xW1))W2
        h = self.w1(x)
        return self.w2(h * F.silu(h))


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


@contextmanager
def mid_layer_inject_train(model, layer_index: int, inject_state: torch.Tensor, scale: float):
    layers = None
    for root in (model, getattr(model, "model", None), getattr(model, "base_model", None)):
        if root is None:
            continue
        if hasattr(root, "layers"):
            layers = root.layers
            break
        inner = getattr(root, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            layers = inner.layers
            break
    if layers is None:
        raise RuntimeError("Unable to locate decoder layers for mid-layer injection.")
    if layer_index < 0 or layer_index >= len(layers):
        raise RuntimeError(f"mid-layer index out of range: {layer_index} (layers={len(layers)})")

    def _hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
            add = inject_state.to(device=hidden.device, dtype=hidden.dtype)
            if add.dim() == 2:
                add = add.unsqueeze(1)
            hidden = hidden.clone()
            hidden[:, -1:, :] = hidden[:, -1:, :] + (float(scale) * add[:, -1:, :])
            return (hidden, *rest)
        hidden = output
        add = inject_state.to(device=hidden.device, dtype=hidden.dtype)
        if add.dim() == 2:
            add = add.unsqueeze(1)
        hidden = hidden.clone()
        hidden[:, -1:, :] = hidden[:, -1:, :] + (float(scale) * add[:, -1:, :])
        return hidden

    handle = layers[layer_index].register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


def normalize_answer(text: str) -> str:
    lowered = text.strip().lower().replace("in the ", "").replace("the ", "")
    return NON_ALNUM_RE.sub("", lowered)


def canonicalize_roles(text: str) -> str:
    found = {}
    for person, role in ROLE_RE.findall(text):
        found[person.lower()] = role.lower()
    if {"a", "b", "c"}.issubset(found.keys()):
        return f"a={found['a']},b={found['b']},c={found['c']}"
    return ""


def answers_match(expected: str, predicted: str) -> bool:
    n_expected = normalize_answer(expected)
    n_pred = normalize_answer(predicted)
    if "a=knight,b=knave,c=knight" in n_expected or "a=knight,b=knight,c=knave" in n_expected:
        c_expected = canonicalize_roles(expected)
        c_pred = canonicalize_roles(predicted)
        if c_expected and c_pred:
            return c_expected == c_pred
    return n_pred.startswith(n_expected) or (n_expected == n_pred)


def extract_answer(text: str) -> str:
    if "ANSWER:" in text:
        tail = text.split("ANSWER:")[-1].strip()
        return tail.splitlines()[0].strip() if tail else ""
    if "Final answer:" in text:
        tail = text.split("Final answer:")[-1].strip()
        return tail.splitlines()[0].strip() if tail else ""
    lines = text.strip().splitlines()
    return lines[-1].strip() if lines else ""


def build_logic_prompt(question: str) -> str:
    return (
        "You are a rigid symbolic reasoner.\n"
        "Output a concise symbolic TRACE and then ANSWER.\n\n"
        f"QUESTION: {question}\n"
        "TRACE:"
    )


def build_final_prompt_prefix(question: str) -> str:
    return (
        "Solve the logic question. Return only the final answer with no explanation.\n\n"
        f"Question: {question}"
    )


def run_b_answer(model, tokenizer, question: str, max_new_tokens: int) -> str:
    prompt = build_final_prompt_prefix(question) + "\nFinal answer:"
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return extract_answer(tokenizer.decode(out[0][ids.shape[1] :], skip_special_tokens=True))


def extract_last_hidden(model, tokenizer, question: str, max_logic_new_tokens: int) -> torch.Tensor:
    logic_prompt = build_logic_prompt(question)
    start_ids = tokenizer(logic_prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model(
            input_ids=start_ids,
            attention_mask=torch.ones_like(start_ids, device=model.device),
            use_cache=True,
            return_dict=True,
            output_hidden_states=True,
        )
    past = out.past_key_values
    tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    last_hidden = out.hidden_states[-1][:, -1:, :]
    cur_len = start_ids.shape[1] + 1
    if tokenizer.eos_token_id is not None and int(tok.item()) == int(tokenizer.eos_token_id):
        return last_hidden
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
        last_hidden = out.hidden_states[-1][:, -1:, :]
        cur_len += 1
        if tokenizer.eos_token_id is not None and int(tok.item()) == int(tokenizer.eos_token_id):
            break
    return last_hidden


def collect_run_b_success_pairs(
    model,
    tokenizer,
    dataset_size: int,
    seeds: List[int],
    sample_size: int,
    max_new_tokens: int,
    max_examples: int,
) -> List[TrainItem]:
    rows: List[TrainItem] = []
    for seed in seeds:
        dataset = generate_dataset(size=dataset_size, seed=seed)
        _, _, test = split_dataset(dataset)
        for p in test[:sample_size]:
            pred = run_b_answer(model, tokenizer, p.prompt, max_new_tokens=max_new_tokens)
            if answers_match(p.answer, pred):
                rows.append(TrainItem(prompt=p.prompt, target_answer=pred))
            if len(rows) >= max_examples:
                return rows
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SwiGLU mid-layer bridge (H3 micro-SFT).")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--sample-size", type=int, default=24)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 11])
    p.add_argument("--dataset-size", type=int, default=1000)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--max-final-new-tokens", type=int, default=48)
    p.add_argument("--target-answer-tokens", type=int, default=6)
    p.add_argument("--max-train-examples", type=int, default=64)
    p.add_argument("--train-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--expansion-factor", type=int, default=2)
    p.add_argument("--layer-index", type=int, default=12)
    p.add_argument("--layer-scale", type=float, default=1.0)
    p.add_argument("--handoff-suffix", type=str, default="\nTherefore, the final answer is ")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-bridge", type=Path, default=Path("runs/projections/swiglu_midlayer_bridge.pt"))
    p.add_argument("--output-report", type=Path, default=Path("runs/swiglu_bridge_report.json"))
    p.add_argument("--run-h3-eval", action="store_true")
    p.add_argument("--h-series-output-root", type=Path, default=Path("runs/true_coconut_h_series"))
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("Missing dependency: peft. Install with `pip install peft`.") from exc

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

    train_items = collect_run_b_success_pairs(
        model=model,
        tokenizer=tokenizer,
        dataset_size=args.dataset_size,
        seeds=args.seeds,
        sample_size=args.sample_size,
        max_new_tokens=args.max_final_new_tokens,
        max_examples=args.max_train_examples,
    )
    if not train_items:
        raise RuntimeError("No successful Run-B examples found for SwiGLU bridge training.")

    hidden_size = int(model.config.hidden_size)
    bridge = SwiGLUBridge(hidden_size=hidden_size, expansion_factor=args.expansion_factor).to(model.device, dtype=model.dtype)
    bridge.train()
    opt = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    emb_layer = model.get_input_embeddings()
    loss_history: List[float] = []
    step_losses: List[float] = []
    for step in range(args.train_steps):
        item = train_items[step % len(train_items)]
        with torch.no_grad():
            h_t = extract_last_hidden(model, tokenizer, item.prompt, max_logic_new_tokens=args.max_logic_new_tokens).to(model.dtype)

        prefix = build_final_prompt_prefix(item.prompt) + args.handoff_suffix
        prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(model.device)
        target_ids = tokenizer(" " + item.target_answer, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        if int(target_ids.numel()) == 0:
            continue
        target_ids = target_ids[:, : args.target_answer_tokens]
        if int(target_ids.shape[1]) == 0:
            continue

        injected = bridge(h_t)
        opt.zero_grad(set_to_none=True)
        with adapter_disabled(model):
            prefix_emb = emb_layer(prefix_ids)
            attn = torch.ones((1, prefix_emb.shape[1]), dtype=torch.long, device=model.device)
            with mid_layer_inject_train(model, layer_index=args.layer_index, inject_state=injected, scale=args.layer_scale):
                out = model(
                    inputs_embeds=prefix_emb,
                    attention_mask=attn,
                    use_cache=True,
                    return_dict=True,
                )
                logits = out.logits[:, -1, :]
                past = out.past_key_values
                cur_len = int(prefix_emb.shape[1]) + 1
                loss = F.cross_entropy(logits, target_ids[:, 0])
                tok = target_ids[:, 0:1]

                for t in range(1, int(target_ids.shape[1])):
                    am = torch.ones((1, cur_len), dtype=torch.long, device=model.device)
                    out = model(
                        input_ids=tok,
                        attention_mask=am,
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                    )
                    logits = out.logits[:, -1, :]
                    past = out.past_key_values
                    loss = loss + F.cross_entropy(logits, target_ids[:, t])
                    tok = target_ids[:, t : t + 1]
                    cur_len += 1
                loss = loss / int(target_ids.shape[1])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), max_norm=1.0)
        opt.step()
        step_loss = float(loss.item())
        step_losses.append(step_loss)
        if (step + 1) % 25 == 0:
            loss_history.append(sum(step_losses) / len(step_losses))
            step_losses = []

    if step_losses:
        loss_history.append(sum(step_losses) / len(step_losses))

    args.output_bridge.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": bridge.state_dict(),
            "expansion_factor": args.expansion_factor,
            "hidden_size": hidden_size,
            "layer_index": args.layer_index,
            "layer_scale": args.layer_scale,
            "train_steps": args.train_steps,
            "train_examples": len(train_items),
        },
        args.output_bridge,
    )

    report: Dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "train_examples": len(train_items),
        "train_steps": args.train_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "expansion_factor": args.expansion_factor,
        "layer_index": args.layer_index,
        "layer_scale": args.layer_scale,
        "loss_history": loss_history,
        "bridge_path": str(args.output_bridge),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.run_h3_eval:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "run_true_coconut_h_series.py"),
            "--base-model",
            args.base_model,
            "--adapter",
            str(args.adapter),
            "--h3-adapter",
            str(args.adapter),
            "--h3-bridge",
            str(args.output_bridge),
            "--h3-layer-index",
            str(args.layer_index),
            "--h3-layer-scale",
            str(args.layer_scale),
            "--sample-size",
            str(args.sample_size),
            "--seeds",
            *[str(s) for s in args.seeds],
            "--dataset-size",
            str(args.dataset_size),
            "--max-logic-new-tokens",
            str(args.max_logic_new_tokens),
            "--max-final-new-tokens",
            str(args.max_final_new_tokens),
            "--only-runs",
            "H3",
            "--output-root",
            str(args.h_series_output_root),
            "--execute",
        ]
        if args.local_files_only:
            cmd.append("--local-files-only")
        subprocess.call(cmd)

    print(f"Wrote bridge: {args.output_bridge}")
    print(f"Wrote report: {args.output_report}")
    print(f"train_examples: {len(train_items)}")
    print(f"loss_history: {loss_history}")


if __name__ == "__main__":
    main()
