# H5 Ablation Extension

This document defines the lightweight contract for H5 ablation extensions layered on top of the A-E matrix.

## Manifest Additions

Add an `h5_extensions` array to the ablation manifest.

Required run rows:

- `H5-PROV`
- `H5-OOD`
- `H5-DPTR`

Each row must include:

- `run_id`
- `name`
- `status`
- `return_code`
- `output`
- `metrics`
- `notes`

Example:

```json
{
  "h5_extensions": [
    {
      "run_id": "H5-PROV",
      "name": "H5 Provenance",
      "status": "ok",
      "return_code": 0,
      "output": "runs/coconut_ablation_matrix/h5_prov.json",
      "metrics": {
        "final_acc": 0.39
      },
      "notes": "Lineage and grounding checks."
    },
    {
      "run_id": "H5-OOD",
      "name": "H5 Out-of-Distribution",
      "status": "ok",
      "return_code": 0,
      "output": "runs/coconut_ablation_matrix/h5_ood.json",
      "metrics": {
        "final_acc": 0.31
      },
      "notes": "Distribution-shift stress test."
    },
    {
      "run_id": "H5-DPTR",
      "name": "H5 Distillation Pointer Transfer",
      "status": "ok",
      "return_code": 0,
      "output": "runs/coconut_ablation_matrix/h5_dptr.json",
      "metrics": {
        "final_acc": 0.36
      },
      "notes": "Pointer transfer/distillation check."
    }
  ]
}
```

## Markdown Summary Rows

When these rows are present in the manifest, the markdown summary must include:

- A row for `H5-PROV`
- A row for `H5-OOD`
- A row for `H5-DPTR`
