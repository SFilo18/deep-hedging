import torch
import math

import config

def gbm(cfg, n_paths, generator=None):

    rands = torch.randn(n_paths, cfg.N, device = cfg.device, dtype= cfg.dtype, generator = generator)

    paths = rands*math.sqrt(cfg.dt)*cfg.sigma + (cfg.mu-.5*cfg.sigma**2)*cfg.dt
    paths = torch.cumsum(paths, 1)
    paths = torch.cat((paths.new_zeros(n_paths,1), paths), dim =1)
    paths = cfg.S0*torch.exp(paths)

    return paths


