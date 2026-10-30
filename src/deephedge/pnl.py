import torch

def hedge_pnl(paths, deltas, premium, K, cost, return_components=False):
    
    assert deltas.shape == (paths.shape[0], paths.shape[1] - 1)

    payoff = (paths[:,-1]-K).clamp_min(0)
    hedge= (torch.diff(paths, dim=1)*deltas).sum(dim=1)

    zero_col = deltas.new_zeros(paths.shape[0],1)
    trades = torch.diff(deltas, dim=1, prepend=zero_col, append=zero_col)

    #add return components dict
    tcosts = cost*((trades.abs()*paths).sum(dim=1))

    return premium - payoff + hedge -tcosts
