from torchvision import models


def get_model(model_arch, num_labels, pretrained=True):
    model = models.segmentation.segmentation._segm_resnet(
        'deeplab', 'resnet50', num_labels, False, pretrained_backbone=True)
    return model
