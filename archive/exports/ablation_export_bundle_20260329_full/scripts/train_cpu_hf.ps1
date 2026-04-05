param(
  [string]$PythonExe = "C:\Users\Andrew\miniconda3\envs\belief\python.exe",
  [string]$BaseModel = "C:\Users\Andrew\hf_models\Qwen2.5-0.5B-Instruct",
  [string]$Dataset = "D:\lojbanhypop\runs\lora_sft_dataset.jsonl",
  [string]$OutputDir = "C:\Users\Andrew\lm_runs\lora_qwen25_05b_symbolic_cpu",
  [int]$Epochs = 1,
  [int]$BatchSize = 1,
  [int]$GradAccum = 4,
  [int]$MaxLength = 256
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "D:\lojbanhypop\src"
$env:CUDA_VISIBLE_DEVICES = ""
$env:PYTHONIOENCODING = "utf-8"

& $PythonExe "D:\lojbanhypop\scripts\train_lora.py" `
  --base-model $BaseModel `
  --dataset $Dataset `
  --output-dir $OutputDir `
  --epochs $Epochs `
  --per-device-batch-size $BatchSize `
  --grad-accum $GradAccum `
  --max-length $MaxLength `
  --logging-steps 1 `
  --save-steps 10 `
  --bnb-4bit-compute-dtype float32

if ($LASTEXITCODE -ne 0) {
  throw "CPU HF training failed with exit code $LASTEXITCODE"
}

Write-Host "CPU HF training complete: $OutputDir"
