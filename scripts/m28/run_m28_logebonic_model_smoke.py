from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lojban_evolution.m28.family import M28_REGISTRY  # noqa: E402
from lojban_evolution.m28.model import train_logebonic_symbiote_model  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and checkpoint the M28 actual Logebonic Symbiote model.")
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--eval-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--max-frames", type=int, default=6)
    parser.add_argument("--max-symbols", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--advisor-hidden-dim", type=int, default=32)
    parser.add_argument("--symbol-budget", type=int, default=0)
    parser.add_argument("--disable-relevance-runtime", action="store_true")
    parser.add_argument("--relevance-rank-weight", type=float, default=0.25)
    parser.add_argument("--disable-relevance-answer", action="store_true")
    parser.add_argument("--run-baselines", action="store_true")
    parser.add_argument("--baseline-epochs", type=int, default=2)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=0)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = Path(args.output_root or M28_REGISTRY["M28"]["output_roots"]["model"])
    result = train_logebonic_symbiote_model(
        train_size=int(args.train_size),
        eval_size=int(args.eval_size),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
        max_frames=int(args.max_frames),
        max_symbols=int(args.max_symbols),
        embedding_dim=int(args.embedding_dim),
        hidden_dim=int(args.hidden_dim),
        advisor_hidden_dim=int(args.advisor_hidden_dim),
        symbol_budget=int(args.symbol_budget),
        enable_relevance_runtime=not bool(args.disable_relevance_runtime),
        relevance_rank_weight=float(args.relevance_rank_weight),
        use_relevance_answer=not bool(args.disable_relevance_answer),
        run_baselines=bool(args.run_baselines),
        baseline_epochs=int(args.baseline_epochs),
        resume_checkpoint=args.resume_checkpoint,
        checkpoint_every_epochs=int(args.checkpoint_every_epochs),
        use_amp=bool(args.use_amp),
        device=str(args.device),
        output_root=output_root,
        run_id=args.run_id,
    )
    print(f"M28 Logebonic Symbiote report written to {result.report_path}")
    print(f"M28 Logebonic Symbiote checkpoint written to {result.checkpoint_path}")
    print(json.dumps(result.metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
