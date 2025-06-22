import torch
import torch.nn as nn


class ConvNeXtBlock(nn.Module):
    def __init__(self,dim):
        super(ConvNeXtBlock, self).__init__()
        self.dwconv=nn.Conv2d(dim,dim,7,padding=3,groups=dim)
        self.norm=nn.LayerNorm(dim,eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act=nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
    
    def forward(self,x):
        inputx=x
        x=self.dwconv(x)
        x=x.permute(0,2,3,1) #[B,H,W,C]
        x=self.norm(x)
        x=self.pwconv1(x)
        x=self.act(x)
        x=self.pwconv2(x)
        x=x.permute(0,3,1,2) #[B,C,H,W]
        return x+inputx
