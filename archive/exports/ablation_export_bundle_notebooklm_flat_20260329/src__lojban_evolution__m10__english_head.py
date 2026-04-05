import torch
import torch.nn as nn
import torch.nn.functional as F

class M10cEnglishHead(nn.Module):
    """
    M10c: English-only Answer Head.
    Physically forbids Lojban tokens by projecting ONLY into the answer class space.
    Input: Logic-conditioned hidden state [B, H]
    Output: Classification over the 14 target English answer tokens.
    """
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, h_final: torch.Tensor) -> torch.Tensor:
        # h_final: [B, 1, H] -> [B, H]
        return self.net(h_final.squeeze(1))
