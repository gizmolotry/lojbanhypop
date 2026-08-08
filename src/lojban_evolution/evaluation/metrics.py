import torch

def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if logits.size(0) == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    return float(preds.eq(targets).float().mean().item())
