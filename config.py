from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class Config:
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype: torch.dtype = torch.float32
    S0: float = 100.0
    K: float = 100.0
    sigma: float = 0.2
    mu: float = 0.0
    T: float = 30 / 365
    N: int = 30
    cost: float = 0.0
    seed_test: int = 67

    @property
    def dt(self) -> float:
        return self.T / self.N
