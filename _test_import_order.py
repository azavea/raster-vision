import sys
d1 = sys.argv[1]
d2 = sys.argv[2]


def make_import_str(dep):
    s = f'import {dep}'
    if dep == 'gdal':
        s = 'from osgeo import gdal'
    elif dep == 'rastervision':
        s = 'from rastervision.core.data import RasterioSource'
    return s


exec(make_import_str(d1))
if d2 != d1:
    exec(make_import_str(d2))

from torchvision.models import resnet50
model = resnet50(pretrained=True)
model.to('cuda')
