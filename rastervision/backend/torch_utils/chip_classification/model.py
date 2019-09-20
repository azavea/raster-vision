from torchvision import models
from torch import nn


def get_model(model_arch, num_labels, pretrained=True):
    model = getattr(models, model_arch)(pretrained=True, progress=True)
    model.fc = nn.Linear(model.fc.in_features, num_labels)
    return model
