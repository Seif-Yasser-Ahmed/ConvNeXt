import torch
from convnext_model import ConvNeXt

model = ConvNeXt(num_classes=10)
x = torch.randn(1, 3, 224, 224)
logits = model(x)
print(logits.shape)  # torch.Size([1, 10])