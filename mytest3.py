import torch

from torchvision.models import resnet18
m = resnet18()

with torch.inference_mode():
    m(torch.zeros((1, 3, 256, 256)))

from torchvision.models import resnet18
m = resnet18()
