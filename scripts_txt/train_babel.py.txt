from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

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
    answer: str
    logic_trace: str


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
    stripped = text.strip()
    lines = stripped.splitlines()
    return lines[-1].strip() if lines else ""


def build_prompt(question: str, style: str) -> str:
    if style == "symbolic":
        return (
            "You are a rigid symbolic reasoner.\n"
            "Output must contain a symbolic TRACE line and an ANSWER line.\n\n"
            f"QUESTION: {question}\n"
            "TRACE:"
        )
    return (
        "Solve the logic question. Return only the final answer with no explanation.\n\n"
        f"Question: {question}\n"
        "Final answer:"
    )


def _past_len(past_key_values) -> int:
    if past_key_values is None:
        return 0
    return int(past_key_values[0][0].shape[-2])


def _cache_decode(
    model,
    start_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int],
    past_key_values=None,
):
    assert start_ids.shape[0] == 1, "Only batch_size=1 supported."
    device = start_ids.device
    current_past = past_key_values
    generated = []

    if current_past is None:
        am = torch.ones_like(start_ids, device=device)
        with torch.no_grad():
            out = model(
                input_ids=start_ids,
                attention_mask=am,
                use_cache=True,
                return_dict=True,
            )
        current_past = out.past_key_values
        tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(tok)
        cur_len = start_ids.shape[1] + 1
        if eos_token_id is not None and int(tok.item()) == int(eos_token_id):
            return torch.cat(generated, dim=1), current_past
    else:
        cur_len = _past_len(current_past)
        for i in range(int(start_ids.shape[1])):
            tok = start_ids[:, i : i + 1]
            am = torch.ones((1, cur_len + 1), dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(
                    input_ids=tok,
                    attention_mask=am,
                    past_key_values=current_past,
                    use_cache=True,
                    return_dict=True,
                )
            current_past = out.past_key_values
            cur_len += 1
        tok = start_ids[:, -1:]

    for _ in range(max_new_tokens - len(generated)):
        am = torch.ones((1, cur_len + 1), dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(
                input_ids=tok,
                attention_mask=am,
                past_key_values=current_past,
                use_cache=True,
                return_dict=True,
            )
        current_past = out.past_key_values
        tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(tok)
        cur_len += 1
        if eos_token_id is not None and int(tok.item()) == int(eos_token_id):
            break

    if not generated:
        return start_ids.new_empty((1, 0)), current_past
    return torch.cat(generated, dim=1), current_past


def _apply_kv_projection(past_key_values, proj: torch.nn.Linear):
    out = []
    for layer in past_key_values:
        k, v = layer[0], layer[1]
        b, h, s, d = k.shape
        k2 = proj(k.reshape(b * h * s, d)).reshape(b, h, s, d)
        v2 = proj(v.reshape(b * h * s, d)).reshape(b, h, s, d)
        if len(layer) > 2:
            out.append((k2, v2, *layer[2:]))
        else:
            out.append((k2, v2))
    return tuple(out)


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    new_ids, _ = _cache_decode(model, ids, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id)
    return extract_answer(tokenizer.decode(new_ids[0], skip_special_tokens=True))


def _load_ablation(ablation_json: Path) -> dict:
    return json.loads(ablation_json.read_text(encoding="utf-8"))


def collect_success_examples(
    model,
    tokenizer,
    dataset_size: int,
    seeds: List[int],
    sample_size: int,
    max_new_tokens: int,
    max_examples: int,
) -> List[TrainItem]:
    out: List[TrainItem] = []
    for seed in seeds:
        dataset = generate_dataset(size=dataset_size, seed=seed)
        _, _, test = split_dataset(dataset)
        for p in test[:sample_size]:
            final_prompt = build_prompt(p.prompt, style="final_answer")
            pred = generate_text(model, tokenizer, final_prompt, max_new_tokens=max_new_tokens)
            if not answers_match(p.answer, pred):
                continue
            logic_prompt = build_prompt(p.prompt, style="symbolic")
            logic_trace = generate_text(model, tokenizer, logic_prompt, max_new_tokens=max_new_tokens)
            out.append(TrainItem(prompt=p.prompt, answer=p.answer, logic_trace=logic_trace))
            if len(out) >= max_examples:
                return out
    return out


def _append_ablation_md(path: Path, line: str) -> None:
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    marker = "## Trinity Expansion"
    if marker not in content:
        content = content.rstrip() + "\n\n## Trinity Expansion\n\n"
    content = content.rstrip() + f"\n- {line}\n"
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a 1-layer Babel bridge projection for latent handoff.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--ablation-json", type=Path, required=True)
    p.add_argument("--output-projection", type=Path, default=Path("runs/projections/babel_bridge.pt"))
    p.add_argument("--output-report", type=Path, default=Path("runs/babel_bridge_report.json"))
    p.add_argument("--ablation-md", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-train-examples", type=int, default=64)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--target-answer-tokens", type=int, default=3)
    p.add_argument("--eval-output", type=Path, default=None)
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("Missing dependency: peft. Install with `pip install peft`.") from exc

    meta = _load_ablation(args.ablation_json)
    seeds = [int(x) for x in meta.get("seeds", [7, 11])]
    sample_size = int(meta.get("sample_size", 24))
    dataset_size = int(meta.get("dataset_size", 1000))
    max_new_tokens = int(meta.get("max_new_tokens", 48))

    adapter_has_tokenizer = (args.adapter / "tokenizer.json").exists() or (args.adapter / "tokenizer_config.json").exists()
    tokenizer_src = str(args.adapter) if adapter_has_tokenizer else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, str(args.adapter), local_files_only=args.local_files_only)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    train_items = collect_success_examples(
        model=model,
        tokenizer=tokenizer,
        dataset_size=dataset_size,
        seeds=seeds,
        sample_size=sample_size,
        max_new_tokens=max_new_tokens,
        max_examples=args.max_train_examples,
    )
    if not train_items:
        raise RuntimeError("No successful Run-B examples found for Babel bridge training.")

    device = next(model.parameters()).device
    # Projection must operate in KV head-dim space to be compatible with Run-E handoff evaluator.
    probe_ids = tokenizer("probe", return_tensors="pt").input_ids.to(device)
    _, probe_past = _cache_decode(model, probe_ids, max_new_tokens=1, eos_token_id=tokenizer.eos_token_id, past_key_values=None)
    head_dim = int(probe_past[0][0].shape[-1])
    bridge = torch.nn.Linear(head_dim, head_dim, bias=False).to(device)
    torch.nn.init.eye_(bridge.weight)
    opt = torch.optim.AdamW(bridge.parameters(), lr=args.lr)

    loss_history: List[float] = []
    for _epoch in range(args.epochs):
        epoch_losses: List[float] = []
        for item in train_items:
            logic_prompt = build_prompt(item.prompt, style="symbolic")
            logic_ids = tokenizer(logic_prompt, return_tensors="pt").input_ids.to(device)
            _, logic_past = _cache_decode(
                model,
                logic_ids,
                max_new_tokens=args.max_logic_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                past_key_values=None,
            )

            suffix_ids = tokenizer("\nTherefore, the final answer is ", return_tensors="pt").input_ids.to(device)
            target_ids = tokenizer(" " + item.answer, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            if target_ids.numel() == 0:
                continue
            target_ids = target_ids[:, : args.target_answer_tokens]
            current_past = logic_past
            cur_len = _past_len(current_past)

            # Ingest suffix into cache before first supervised answer token.
            with adapter_disabled(model):
                for i in range(int(suffix_ids.shape[1])):
                    tok = suffix_ids[:, i : i + 1]
                    am = torch.ones((1, cur_len + 1), dtype=torch.long, device=device)
                    proj_past = _apply_kv_projection(current_past, bridge)
                    out = model(
                        input_ids=tok,
                        attention_mask=am,
                        past_key_values=proj_past,
                        use_cache=True,
                        return_dict=True,
                    )
                    current_past = out.past_key_values
                    cur_len += 1

            # Teacher-force first N answer tokens while optimizing only bridge.
            seq_loss = 0.0
            tok = suffix_ids[:, -1:]
            steps = int(target_ids.shape[1])
            opt.zero_grad(set_to_none=True)
            with adapter_disabled(model):
                for t in range(steps):
                    am = torch.ones((1, cur_len + 1), dtype=torch.long, device=device)
                    proj_past = _apply_kv_projection(current_past, bridge)
                    out = model(
                        input_ids=tok,
                        attention_mask=am,
                        past_key_values=proj_past,
                        use_cache=True,
                        return_dict=True,
                    )
                    current_past = out.past_key_values
                    cur_len += 1
                    logits = out.logits[:, -1, :]
                    tgt = target_ids[:, t]
                    seq_loss = seq_loss + F.cross_entropy(logits, tgt)
                    tok = tgt.unsqueeze(1)

            loss = seq_loss / max(1, steps)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.item()))
        loss_history.append(sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0)

    args.output_projection.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": bridge.weight.detach().cpu()}, args.output_projection)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "ablation_json": str(args.ablation_json),
        "train_examples": len(train_items),
        "epochs": args.epochs,
        "lr": args.lr,
        "projection_dim": head_dim,
        "loss_history": loss_history,
        "projection_path": str(args.output_projection),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    eval_output = args.eval_output
    if eval_output is not None:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "eval_hf_dual_mode_handoff.py"),
            "--base-model",
            args.base_model,
            "--adapter",
            str(args.adapter),
            "--sample-size",
            str(sample_size),
            "--seeds",
            *[str(s) for s in seeds],
            "--dataset-size",
            str(dataset_size),
            "--max-new-tokens",
            str(max_new_tokens),
            "--handoff-projection",
            str(args.output_projection),
            "--output",
            str(eval_output),
        ]
        if args.local_files_only:
            cmd.append("--local-files-only")
        subprocess.call(cmd)

    if args.ablation_md is not None:
        line = f"`Run E (Babel)` bridge trained on {len(train_items)} examples; projection: `{args.output_projection}`."
        _append_ablation_md(args.ablation_md, line)

    print(f"Wrote projection: {args.output_projection}")
    print(f"Wrote report: {args.output_report}")
    print(f"train_examples: {len(train_items)}")
    print(f"loss_history: {loss_history}")


if __name__ == "__main__":
    main()
