'''
from osgeo import gdal
import shapely
import geopandas
import numpy
import PIL
import pyproj
import sklearn
import scipy
import cv2
import imageio
import tensorboard
import albumentations
import cython
import pycocotools
import matplotlib
import numpy
'''

import numpy
from rastervision.core.data import RasterioSource
from torchvision.models import resnet50

model = resnet50(pretrained=True)
model.to('cuda')

# segfaults:
# import rv, torch
# import [shapely | numpy | PIL | scipy | imageio | tensorboard | cython | pycocotools | numpy], rv, torch

# no segfault:
# import torch
# import [gdal | geopandas | pyproj | sklearn | cv2 | albumentations | matplotlib], rv, torch
