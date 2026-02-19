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

Outputs are written to `runs/<timestamp>/`:

- `history.json`
- `summary.md`
