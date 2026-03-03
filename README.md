# Lojban Hypothesis - Identity Tracking Phase

This project runs an iterative language-evolution loop on topologically tangled reasoning tasks:

- Winograd-style pronoun resolution
- Multi-agent belief state tracking
- Knights/knaves relational logic graphs

The goal is to pressure the system toward low-cost identity pointers, not arithmetic compression.

## Run

```powershell
$env:PYTHONPATH="src"
python scripts/run_experiment.py --iterations 6 --dataset-size 1000
```

Outputs are written to `artifacts/runs/<timestamp>/`:

- `history.json`
- `summary.md`
- `run_manifest.json`

One-command task entrypoint:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\tasks.ps1 -Task run
```

## Airflow Orchestration

Airflow DAG wrappers are available for the canonical train/eval pipelines:

- `airflow/dags/lojban_experiment_dag.py` (`scripts/pipeline_train_grounded_reasoner.py`)
- `airflow/dags/lojban_phase_ablation_dag.py` (`scripts/pipeline_eval_manifold.py`)

Setup and runtime configuration are documented in:

- `docs/AIRFLOW_ORCHESTRATION.md`

Artifact-contract-first partitions for S3-backed orchestration:

- `models/frozen_manifolds` for train outputs
- `telemetry/raw` for eval outputs

Artifact contract schema:

- `docs/artifact_contract_v1.schema.json`

## Phase 4: LoRA Weight Mutation

Build an SFT dataset from Phase 3 successful runs:

```powershell
$env:PYTHONPATH="src"
python scripts/build_lora_dataset.py \
  --eval-json C:\Users\Andrew\lm_runs\lm_eval_20260219_221552.json \
             C:\Users\Andrew\lm_runs\lm_eval_20260219_222027.json \
  --mode phase3_fewshot \
  --output runs/lora_sft_dataset.jsonl
```

Train a LoRA adapter (PEFT) on the symbolic traces:

```powershell
$env:PYTHONPATH="src"
python scripts/train_lora.py \
  --base-model Qwen/Qwen2.5-Coder-14B-Instruct \
  --dataset runs/lora_sft_dataset.jsonl \
  --output-dir runs/lora_qwen25_14b_symbolic
```

Notes:
- `build_lora_dataset.py` reconstructs canonical symbolic traces from `problem_id` and dataset seed, then pairs them with gold answers.
- `train_lora.py` trains LoRA adapters only (base model weights remain frozen).
- Required packages: `transformers`, `datasets`, `peft`, `accelerate`, `torch`.

### Offline / Local Model Path

If internet is blocked, pass a local Transformers checkpoint directory as `--base-model` and keep `--local-files-only`.

Example (cached phi-2):

```powershell
$env:PYTHONPATH="src"
C:\Users\Andrew\miniconda3\envs\belief\python.exe scripts\train_lora.py \
  --base-model C:\Users\Andrew\.cache\huggingface\hub\models--microsoft--phi-2\snapshots\ef382358ec9e382308935a992d908de099b64c23 \
  --dataset runs/lora_sft_dataset.jsonl \
  --output-dir C:\Users\Andrew\lm_runs\lora_phi2_symbolic \
  --local-files-only
```

For Qwen LoRA in this offline setup, use a local **Transformers-format** Qwen checkpoint path (not GGUF).

## GGUF-Compatible Mode (LM Studio)

For GGUF-only models, use a reusable reasoning pack (examples + rigid rules) and inject it at inference time.

Build the pack from successful eval traces:

```powershell
$env:PYTHONPATH="src"
python scripts/build_gguf_pack.py \
  --eval-json C:\Users\Andrew\lm_runs\lm_eval_20260219_221552.json \
  --mode phase3_fewshot \
  --max-examples 8 \
  --output C:\Users\Andrew\lm_runs\gguf_reasoning_pack_v1.json
```

Evaluate GGUF with the pack:

```powershell
$env:PYTHONPATH="src"
python scripts/eval_with_lms.py \
  --model qwen2.5-coder-14b-instruct \
  --sample-size 24 \
  --modes baseline gguf_pack \
  --pack-file C:\Users\Andrew\lm_runs\gguf_reasoning_pack_v1.json \
  --output-dir C:\Users\Andrew\lm_runs
```

Note: pure GGUF does not support this repo's PEFT training flow directly; this pack mode is the compatible path for LM Studio GGUF inference.

## Qwen 14B QLoRA Training (Works)

This repository now supports 4-bit QLoRA for large models on limited VRAM.

Recommended command (cache on D: to avoid C: disk exhaustion):

```powershell
$env:PYTHONPATH="src"
$env:HF_HOME="D:\hf_cache"
$env:HUGGINGFACE_HUB_CACHE="D:\hf_cache\hub"
C:\Users\Andrew\miniconda3\envs\belief\python.exe scripts\train_lora.py \
  --base-model Qwen/Qwen2.5-Coder-14B-Instruct \
  --dataset runs/lora_sft_dataset.jsonl \
  --output-dir runs/lora_qwen25_14b_symbolic_qlora \
  --epochs 1 \
  --per-device-batch-size 1 \
  --grad-accum 8 \
  --max-length 512 \
  --load-in-4bit \
  --gradient-checkpointing \
  --bnb-4bit-compute-dtype bfloat16
```

## CPU HF One-Command Training

Run CPU-only training with local HF model files:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_cpu_hf.ps1
```

Optional overrides:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_cpu_hf.ps1 \
  -BaseModel "C:\Users\Andrew\hf_models\Qwen2.5-0.5B-Instruct" \
  -OutputDir "C:\Users\Andrew\lm_runs\lora_qwen25_05b_symbolic_cpu"
```

## Evaluate Base vs Adapter (HF)

```powershell
$env:PYTHONPATH="src"
$env:CUDA_VISIBLE_DEVICES=""
C:\Users\Andrew\miniconda3\envs\belief\python.exe scripts\eval_hf_adapter.py \
  --base-model C:\Users\Andrew\hf_models\Qwen2.5-0.5B-Instruct \
  --adapter C:\Users\Andrew\lm_runs\lora_qwen25_05b_symbolic_cpu \
  --sample-size 24 \
  --output C:\Users\Andrew\lm_runs\hf_adapter_eval_qwen25_05b_cpu.json \
  --local-files-only
```

## Full-Phase Ablation + Quarantine

Runs the full six-phase loop (seed -> generate -> meta-analyze -> evaluate -> update -> retrain-proxy),
creates a quarantine snapshot of current code/docs, and compares ablations.

```powershell
$env:PYTHONPATH="src"
C:\Users\Andrew\miniconda3\envs\belief\python.exe scripts\run_phase_ablation.py \
  --dataset-size 1000 \
  --iterations 6 \
  --output-dir artifacts/runs
```

Outputs:
- `artifacts/runs/quarantine_<timestamp>/MANIFEST.json` (frozen snapshot + checksums)
- `artifacts/runs/ablation_<timestamp>/ablation.json`
- `artifacts/runs/ablation_<timestamp>/summary.md`
- `artifacts/runs/ablation_<timestamp>/run_manifest.json`

## Reproducibility + Secrets

- Copy `.env.example` to `.env` for local-only values.
- Primary experiment scripts now emit a `run_manifest.json` including args, git commit, dataset fingerprint, and key environment markers.

## Unified Adapter: Mixed Crystal + Fluid Curriculum

Build a mixed dataset (both symbolic and plain-answer targets) and train one adapter.

One-command CPU path:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_mixed_curriculum_cpu.ps1
```

Manual dataset build:

```powershell
$env:PYTHONPATH="src"
python scripts\build_mixed_curriculum_dataset.py `
  --output C:\Users\Andrew\lm_runs\lora_sft_dataset_mixed_5k.jsonl `
  --dataset-size 1200 `
  --seeds 7 11 13 `
  --copies-per-problem 2 `
  --max-samples 5000
```

## Dual-Mode Gate (Final + Symbolic)

Evaluates one adapter on both prompt styles and enforces minimum lift thresholds.

```powershell
$env:PYTHONPATH="src"
$env:CUDA_VISIBLE_DEVICES=""
C:\Users\Andrew\miniconda3\envs\belief\python.exe scripts\eval_hf_dual_mode_gate.py `
  --base-model C:\Users\Andrew\hf_models\Qwen2.5-0.5B-Instruct `
  --adapter C:\Users\Andrew\lm_runs\lora_qwen25_05b_unified_mixed_cpu `
  --sample-size 48 `
  --seeds 7 11 `
  --min-final-lift 0.05 `
  --min-symbolic-lift 0.20 `
  --output C:\Users\Andrew\lm_runs\hf_dual_mode_gate_unified.json `
  --local-files-only
```

## Coconut Fusion Ablation Matrix (Runs A-E)

Formalizes the five-run thesis test:

- `A` Control: English CoT -> English output (monolithic baseline)
- `B` Rigid Lojban: Phase-5 adapter with text-to-text decoding
- `C` Coconut Fusion: Phase-5 adapter logic + latent KV handoff to adapter-off decode
- `D` NoPE Fusion: DroPE (NoPE) latent handoff
- `E` Babel Bridge: latent handoff with optional linear projection

Plan-only artifact (no execution):

```powershell
$env:PYTHONPATH="src"
python scripts\run_coconut_ablation_matrix.py `
  --base-model C:\Users\Andrew\hf_models\Qwen2.5-0.5B-Instruct `
  --adapter runs\phase5_two_stage_recovery_anchors\20260224_225142\stage2_phase5 `
  --sample-size 24 `
  --seeds 7 11 `
  --dataset-size 1000 `
  --local-files-only
```

Execute all enabled runs:

```powershell
$env:PYTHONPATH="src"
python scripts\run_coconut_ablation_matrix.py `
  --base-model C:\Users\Andrew\hf_models\Qwen2.5-0.5B-Instruct `
  --adapter runs\phase5_two_stage_recovery_anchors\20260224_225142\stage2_phase5 `
  --sample-size 24 `
  --seeds 7 11 `
  --dataset-size 1000 `
  --execute `
  --local-files-only
```

Optional Run D recalibration utility (50-100 update steps with DroPE enabled):

```powershell
$env:PYTHONPATH="src"
python scripts\run_drope_recalibration.py `
  --base-model C:\Users\Andrew\hf_models\Qwen2.5-0.5B-Instruct `
  --dataset runs\lora_sft_dataset.jsonl `
  --adapter-init runs\phase5_two_stage_recovery_anchors\20260224_225142\stage2_phase5 `
  --output-dir runs\drope_recalibrated_adapter `
  --max-steps 64 `
  --local-files-only
```
