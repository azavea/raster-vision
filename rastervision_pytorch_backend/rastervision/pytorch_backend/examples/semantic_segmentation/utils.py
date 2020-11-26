import numpy as np
import albumentations as A

example_rgb_transform = A.Compose([
    A.Flip(),
    A.Transpose(),
    A.RandomRotate90(),
    A.ShiftScaleRotate(),
    A.OneOf([
        A.CLAHE(),
        A.FancyPCA(),
        A.HueSaturationValue(hue_shift_limit=10),
        A.RGBShift(),
        A.ToGray(),
        A.ToSepia(),
    ]),
    A.OneOf([
        A.RandomBrightness(),
        A.RandomGamma(),
    ]),
    A.OneOf([
        A.GaussNoise(),
        A.ISONoise(),
        A.RandomFog(),
    ]),
    A.OneOf([
        A.Blur(),
        A.MotionBlur(),
        A.ImageCompression(),
        A.Downscale(),
    ]),
    A.OneOf([
        A.GridDistortion(),
    ]),
    A.CoarseDropout(max_height=32, max_width=32, max_holes=5)
])

# not all transforms work with more than 3 channels, here are
# some of the ones that do
example_multiband_transform = A.Compose([
    A.Flip(),
    A.Transpose(),
    A.RandomRotate90(),
    A.ShiftScaleRotate(),
    A.FancyPCA(),
    A.GaussNoise(),
    A.OneOf([
        A.Blur(),
        A.MotionBlur(),
        A.Downscale(),
    ]),
    A.OneOf([
        A.GridDistortion(),
    ]),
    A.CoarseDropout(max_height=32, max_width=32, max_holes=5)
])

imagenet_stats = {
    'mean': np.array((0.485, 0.456, 0.406, 0.485)),
    'std': np.array((0.229, 0.224, 0.225, 0.229))
}


def Unnormalize(mean, std):
    return A.Normalize(
        mean=(-mean / std).tolist(),
        std=(1 / std).tolist(),
        max_pixel_value=1.)
