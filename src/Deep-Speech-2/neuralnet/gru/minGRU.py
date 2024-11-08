# Paper: https://arxiv.org/abs/2410.01201v1
# Implementation of minGRU: https://github.com/lucidrains/minGRU-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F


def exists(v):
    return v is not None

def parallel_scan_log(log_coeffs, log_values):
    # log_coeffs: (batch, seq_len, input_size)
    # log_values: (batch, seq_len+1, input_size)
    a_star = log_coeffs.cumsum(dim = 1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim = 1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()


# Appendix: B.3
# Log-Space Version
def g(x):
    return torch.where(x >= 0, x+0.5, torch.sigmoid(x))

def log_g(x):
    return torch.where(x >= 0, (F.relu(x)+0.5).log(), -F.softplus(-x))


# =============================================================================
# MinGRU Implementation
    
class MinGRU(nn.Module):
    def __init__(self, dim, expansion_factor=1.0, dropout_prob=0):
        super().__init__()

        inner_dim = int(dim * expansion_factor)
        self.hidden_gate = nn.Linear(dim, inner_dim * 2, bias=False)
        self.out = nn.Linear(inner_dim, dim, bias=False) if expansion_factor !=1.0 else nn.Identity()
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
        

    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        seq_len = x.shape[1]
        hidden, gate = self.hidden_gate(x).chunk(2, dim=-1)

        hidden = self.dropout(hidden)   # Apply dropout

        log_coeffs = -F.softplus(gate)

        log_z = -F.softplus(-gate)
        log_tilde_h = log_g(hidden)
        log_values = log_z + log_tilde_h
        
        if exists(prev_hidden):
            log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))
            log_values = torch.cat((prev_hidden.log(), log_values), dim=1)

        out = parallel_scan_log(log_coeffs, log_values)
        out = out[:, -seq_len:]  

        next_prev_hidden = out[:, -1]

        out = self.out(out)

        if not return_next_prev_hidden:
            return out
        
        return out, next_prev_hidden


if __name__ == "__main__":
    x = torch.randn(1, 100, 80)
    
    model = MinGRU(dim=512, expansion_factor=2.0, dropout_prob=0.2, bidirectional=False)
    out = model(x)
    print("Output:", out)
    print(out.shape)

    model = MinGRU(dim=512, expansion_factor=2.0, dropout_prob=0.2, bidirectional=True)
    out = model(x)
    print("Output:", out)
    print(out.shape)


