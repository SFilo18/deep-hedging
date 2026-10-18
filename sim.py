import torch

import config

def gbm(cfg, n_paths, generator=None):

    rands = torch.randn(n_paths, cfg.N, cfg.device, cfg.dtype, generator)

    paths = rands*torch.sqrt(cfg.dt)*cfg.sigma + torch.ones_like(rands)*(cfg.mu-.5*cfg.sigma**2)*cfg.dt
    paths = torch.cumsum(paths, 1)
    paths = torch.cat(torch.zeros(n_paths), paths)
    paths = cfg.S0*torch.exp(paths)

    return paths


