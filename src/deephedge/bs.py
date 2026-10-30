import torch
import math

def norm_cdf(x):
    return .5*(1+torch.erf(x/math.sqrt(2)))

def _d1_d2(S, tau, K, sigma):
    tau = tau.clamp_min(1e-12)
    vol_sqrt_tau = sigma*torch.sqrt(tau)

    d1 = (torch.log(S/K)+.5*(sigma**2)*tau)/(vol_sqrt_tau)
    d2=d1-vol_sqrt_tau

    return d1,d2

def bs_call_price(S, tau, K, sigma):
    d1,d2 = _d1_d2(S, tau, K, sigma)
    return S*norm_cdf(d1) - K*norm_cdf(d2)

def bs_call_delta(S,tau,K,sigma):
    d1,_ = _d1_d2(S, tau, K, sigma)
    return norm_cdf(d1)