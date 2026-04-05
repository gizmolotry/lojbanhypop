import torch
import torch.nn as nn
import torch.nn.functional as F

class M10aRecoverabilityProbe(nn.Module):
    """
    M10a: Discriminative Probe.
    Tests if the frozen M9 logic channel carries recoverable answer info.
    Input: Concatenated 10-slot logic tensors [B, 10, 896]
    Output: Classification over target answer tokens.
    """
    def __init__(self, hidden_size: int, num_slots: int, num_classes: int):
        super().__init__()
        self.input_dim = hidden_size * num_slots
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 10, H] -> flatten to [B, 10*H]
        x_flat = x.view(x.shape[0], -1)
        return self.net(x_flat)
