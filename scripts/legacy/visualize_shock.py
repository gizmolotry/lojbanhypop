from __future__ import annotations

import argparse
import json
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from lojban_evolution.experiment import generate_dataset, split_dataset


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


def _past_len(past_key_values) -> int:
    if past_key_values is None:
        return 0
    return int(past_key_values[0][0].shape[-2])


def _cache_decode_with_hidden(model, ids: torch.Tensor, max_new_tokens: int, eos_token_id: int | None):
    device = ids.device
    with torch.no_grad():
        out = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids, device=device),
            use_cache=True,
            return_dict=True,
            output_hidden_states=True,
        )
    cur_past = out.past_key_values
    tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    generated = [tok]
    hidden_per_layer = [h[:, -1, :].detach().cpu() for h in out.hidden_states]
    cur_len = ids.shape[1] + 1
    for _ in range(max_new_tokens - 1):
        am = torch.ones((1, cur_len + 1), dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(
                input_ids=tok,
                attention_mask=am,
                past_key_values=cur_past,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
        cur_past = out.past_key_values
        tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(tok)
        hidden_per_layer = [h[:, -1, :].detach().cpu() for h in out.hidden_states]
        cur_len += 1
        if eos_token_id is not None and int(tok.item()) == int(eos_token_id):
            break
    return torch.cat(generated, dim=1), cur_past, hidden_per_layer


def _handoff_step_hidden_and_attn(model, suffix_ids: torch.Tensor, past):
    cur_past = past
    cur_len = _past_len(cur_past)
    device = suffix_ids.device
    for i in range(int(suffix_ids.shape[1])):
        tok = suffix_ids[:, i : i + 1]
        am = torch.ones((1, cur_len + 1), dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(
                input_ids=tok,
                attention_mask=am,
                past_key_values=cur_past,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
                output_attentions=True,
            )
        cur_past = out.past_key_values
        cur_len += 1
    hidden = [h[:, -1, :].detach().cpu() for h in out.hidden_states]
    # attentions: tuple[layers] each [b, heads, q, k]
    attn = [a[0].detach().cpu() for a in out.attentions]
    return hidden, attn


def _project_2d(vectors: torch.Tensor) -> torch.Tensor:
    try:
        from sklearn.manifold import TSNE

        z = TSNE(n_components=2, init="pca", random_state=7, perplexity=max(2, min(10, vectors.shape[0] - 1))).fit_transform(
            vectors.numpy()
        )
        return torch.tensor(z, dtype=torch.float32)
    except Exception:
        x = vectors - vectors.mean(dim=0, keepdim=True)
        u, s, _ = torch.pca_lowrank(x, q=2)
        return x @ u[:, :2]


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
    p = argparse.ArgumentParser(description="Path 3: quantify latent drift and visualize handoff shock.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--sample-size", type=int, default=24)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dataset-size", type=int, default=1000)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--output-json", type=Path, default=Path("runs/shock_analysis.json"))
    p.add_argument("--output-png", type=Path, default=Path("runs/shock_analysis.png"))
    p.add_argument("--ablation-md", type=Path, default=None)
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Missing dependency: matplotlib. Install with `pip install matplotlib`.") from exc
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("Missing dependency: peft. Install with `pip install peft`.") from exc

    adapter_has_tokenizer = (args.adapter / "tokenizer.json").exists() or (args.adapter / "tokenizer_config.json").exists()
    tokenizer_src = str(args.adapter) if adapter_has_tokenizer else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    base.resize_token_embeddings(len(tokenizer))
    base.eval()
    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    backbone.resize_token_embeddings(len(tokenizer))
    adapted = PeftModel.from_pretrained(backbone, str(args.adapter), local_files_only=args.local_files_only)
    adapted.eval()

    dataset = generate_dataset(size=args.dataset_size, seed=args.seed)
    _, _, test = split_dataset(dataset)
    sample = test[: args.sample_size]

    all_points: List[torch.Tensor] = []
    labels: List[str] = []
    layer_cos: List[torch.Tensor] = []
    handoff_heatmaps: List[torch.Tensor] = []
    for p in sample:
        final_prompt = (
            "Solve the logic question. Return only the final answer with no explanation.\n\n"
            f"Question: {p.prompt}\n"
            "Final answer:"
        )
        logic_prompt = (
            "You are a rigid symbolic reasoner.\n"
            "Output must contain a symbolic TRACE line and an ANSWER line.\n\n"
            f"QUESTION: {p.prompt}\n"
            "TRACE:"
        )
        final_ids = tokenizer(final_prompt, return_tensors="pt").input_ids.to(base.device)
        logic_ids = tokenizer(logic_prompt, return_tensors="pt").input_ids.to(adapted.device)

        with torch.no_grad():
            base_out = base(
                input_ids=final_ids,
                attention_mask=torch.ones_like(final_ids),
                return_dict=True,
                output_hidden_states=True,
            )
        _, logic_past, logic_hidden = _cache_decode_with_hidden(
            adapted, logic_ids, max_new_tokens=args.max_logic_new_tokens, eos_token_id=tokenizer.eos_token_id
        )
        suffix = tokenizer("\nTherefore, the final answer is ", return_tensors="pt").input_ids.to(adapted.device)
        with adapter_disabled(adapted):
            handoff_hidden, attn = _handoff_step_hidden_and_attn(adapted, suffix, logic_past)

        b_layers = [h[:, -1, :].detach().cpu() for h in base_out.hidden_states]
        cos_vals = []
        for bl, ll in zip(b_layers, logic_hidden):
            cos_vals.append(F.cosine_similarity(bl.float(), ll.float(), dim=-1).mean())
        layer_cos.append(torch.stack(cos_vals))

        all_points.append(b_layers[-1].squeeze(0).float())
        labels.append("control_en")
        all_points.append(logic_hidden[-1].squeeze(0).float())
        labels.append("lojban_phase5")
        all_points.append(handoff_hidden[-1].squeeze(0).float())
        labels.append("handoff")

        # Build per-layer attention map as average over heads and query position.
        lay_rows = []
        for layer_attn in attn:
            # layer_attn: [heads, q, k]
            mean_heads = layer_attn.mean(dim=0)  # [q, k]
            lay_rows.append(mean_heads[-1, :])  # last query row
        handoff_heatmaps.append(torch.stack(lay_rows, dim=0))

    points = torch.stack(all_points, dim=0)
    z2 = _project_2d(points)
    layer_cos_t = torch.stack(layer_cos, dim=0) if layer_cos else torch.empty((0, 0))
    mean_layer_cos = layer_cos_t.mean(dim=0) if layer_cos_t.numel() > 0 else torch.empty((0,))
    mean_cos = float(mean_layer_cos.mean().item()) if mean_layer_cos.numel() > 0 else 0.0
    disconnect = mean_cos < 0.3

    avg_heat = None
    if handoff_heatmaps:
        max_k = max(int(h.shape[1]) for h in handoff_heatmaps)
        padded = []
        for h in handoff_heatmaps:
            if int(h.shape[1]) < max_k:
                pad = torch.zeros((h.shape[0], max_k - h.shape[1]), dtype=h.dtype)
                h = torch.cat([h, pad], dim=1)
            padded.append(h)
        avg_heat = torch.stack(padded, dim=0).mean(dim=0)

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    color = {"control_en": "tab:blue", "lojban_phase5": "tab:orange", "handoff": "tab:green"}
    for idx, label in enumerate(labels):
        axes[0].scatter(float(z2[idx, 0]), float(z2[idx, 1]), c=color[label], s=18, alpha=0.75)
    axes[0].set_title("State Clusters (t-SNE/PCA)")
    axes[0].set_xlabel("dim-1")
    axes[0].set_ylabel("dim-2")
    if avg_heat is not None:
        im = axes[1].imshow(avg_heat.numpy(), aspect="auto", interpolation="nearest")
        axes[1].set_title("Handoff Attention Heatmap")
        axes[1].set_xlabel("Key Position")
        axes[1].set_ylabel("Layer")
        fig.colorbar(im, ax=axes[1], shrink=0.8)
    else:
        axes[1].text(0.5, 0.5, "No attention captured", ha="center", va="center")
        axes[1].set_axis_off()
    fig.tight_layout()
    fig.savefig(args.output_png, dpi=150)
    plt.close(fig)

    payload: Dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "dataset_size": args.dataset_size,
        "mean_cosine_similarity": mean_cos,
        "mean_layer_cosine_similarity": mean_layer_cos.tolist() if mean_layer_cos.numel() > 0 else [],
        "disconnect_verdict": "Complete Manifold Disconnect" if disconnect else "Partial/Connected",
        "plot_path": str(args.output_png),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {args.output_json}")
    print(f"Wrote: {args.output_png}")
    print(f"mean cosine similarity: {mean_cos:.4f}")
    if disconnect:
        print("Complete Manifold Disconnect")

    if args.ablation_md is not None:
        _append_ablation_md(
            args.ablation_md,
            f"`Drift Value` mean_cosine={mean_cos:.4f} ({'Complete Manifold Disconnect' if disconnect else 'partial overlap'}).",
        )


if __name__ == "__main__":
    main()
