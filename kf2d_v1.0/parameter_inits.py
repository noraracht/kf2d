import torch
from torch import FloatTensor


# Initializes random float tensor, then wraps it with nn.Parameter

def new_parameter(size, device):

    #out = torch.nn.Parameter(FloatTensor(size), requires_grad=True)
    out = torch.nn.Parameter(torch.ones(size, dtype=torch.float32, requires_grad=True, device = device))
    #torch.nn.init.uniform_(out, 1.0, 1.0)

    return out
