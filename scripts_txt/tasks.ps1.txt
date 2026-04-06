param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("test", "run", "ablation", "smoke")]
  [string]$Task
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

switch ($Task) {
  "test" {
    python -m pytest -q
  }
  "run" {
    python scripts/run_experiment.py --iterations 6 --dataset-size 1000
  }
  "ablation" {
    python scripts/run_phase_ablation.py --dataset-size 1000 --iterations 6
  }
  "smoke" {
    python scripts/run_experiment.py --iterations 1 --dataset-size 60 --max-accept 1
  }
}
