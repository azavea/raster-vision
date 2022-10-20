import torch
from torchvision.models import resnet18

m = resnet18()

with torch.inference_mode(True):
    m = resnet18()
