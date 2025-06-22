import torch
import torch.nn as nn
from convnext_block import ConvNeXtBlock


class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList()
        self.downsample_norms=nn.ModuleList()

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
        )
        self.downsample_layers.append(self.stem)
        self.downsample_norms.append(nn.LayerNorm(dims[0], eps=1e-6))

        for i in range(3):
            self.downsample_layers.append(
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_norms.append(nn.LayerNorm(dims[i+1], eps=1e-6))

        self.stages=nn.ModuleList()
        for i in range(4):
            stage=nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        self.norm=nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head=nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)

    def forward_features(self, x):
        for i in range(4):
            x=self.downsample_layers[i](x)
            x=x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
            x=self.downsample_norms[i](x)
            x=x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
            x=self.stages[i](x)
        return self.norm(x.mean([-2, -1]))
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
