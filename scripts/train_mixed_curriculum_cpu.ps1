param(
  [string]$PythonExe = "C:\Users\Andrew\miniconda3\envs\belief\python.exe",
  [string]$BaseModel = "C:\Users\Andrew\hf_models\Qwen2.5-0.5B-Instruct",
  [string]$DatasetOut = "C:\Users\Andrew\lm_runs\lora_sft_dataset_mixed_5k.jsonl",
  [string]$AdapterOut = "C:\Users\Andrew\lm_runs\lora_qwen25_05b_unified_mixed_cpu",
  [int]$DatasetSize = 1200,
  [int]$MaxSamples = 5000,
  [int]$Epochs = 1,
  [int]$BatchSize = 1,
  [int]$GradAccum = 8,
  [int]$MaxLength = 320
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "D:\lojbanhypop\src"
$env:CUDA_VISIBLE_DEVICES = ""
$env:PYTHONIOENCODING = "utf-8"

& $PythonExe "D:\lojbanhypop\scripts\build_mixed_curriculum_dataset.py" `
  --output $DatasetOut `
  --dataset-size $DatasetSize `
  --seeds 7 11 13 `
  --copies-per-problem 2 `
  --noise-level 2 `
  --max-samples $MaxSamples `
  --fluid-ratio 1.0

if ($LASTEXITCODE -ne 0) {
  throw "Failed to build mixed dataset"
}

& $PythonExe "D:\lojbanhypop\scripts\train_lora.py" `
  --base-model $BaseModel `
  --dataset $DatasetOut `
  --output-dir $AdapterOut `
  --epochs $Epochs `
  --per-device-batch-size $BatchSize `
  --grad-accum $GradAccum `
  --max-length $MaxLength `
  --logging-steps 20 `
  --save-steps 200 `
  --bnb-4bit-compute-dtype float32

if ($LASTEXITCODE -ne 0) {
  throw "Mixed curriculum training failed with exit code $LASTEXITCODE"
}

Write-Host "Unified mixed adapter complete: $AdapterOut"

