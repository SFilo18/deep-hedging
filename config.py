from dataclasses import dataclass
import torch

@dataclass 
class Config:

    S0: float = 100.0
    K: float = 100.0
    sigma : float = 0.2
    mu : float = 0.0
    T : float =30/365
    N : int = 30
    cost : float = 0.0
    device : torch.device
    dtype : torch.float32
    seed_test : int = 67

    dt = T/N
