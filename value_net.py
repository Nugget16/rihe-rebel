# value_net.py
from __future__ import annotations
import torch
import torch.nn as nn
from pbs import PBS_DIM


class ValueNet(nn.Module):
    """
    MLP that maps a 217-dim public-belief state vector to a scalar value
    estimate for the current player.

    Architecture: 217 -> 256 -> 256 -> 1
    """

    def __init__(self, input_dim: int = PBS_DIM, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, PBS_DIM) float tensor
        Returns:
            (batch, 1) value estimates
        """
        return self.net(x)

    def predict(self, pbs_vec, device) -> float:
        x = torch.tensor(pbs_vec, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            return self.forward(x).item()

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, device):
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")