import torch
import torch.nn as nn


class ConvNeXt(nn.Module):
    def __init__(self,dim):
        super(ConvNeXt, self).__init__()
        self.dwconv=nn.Conv2d(dim,dim,7,padding=3,groups=dim)
        self.norm=nn.LayerNorm(dim,eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act=nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
    
    def forward():
        pass
