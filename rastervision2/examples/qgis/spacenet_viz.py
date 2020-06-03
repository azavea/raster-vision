"""Visualize imagery and predictions in QGIS.
This script should be run in the QGIS (3.0+) Python console after
downloading the data locally, and adjusting constants.
"""
from os.path import join, splitext, basename
import glob

from rastervision.utils.files import download_if_needed, file_to_json

def clear_layers():
    layer_ids = QgsProject.instance().mapLayers().keys()
    for layer_id in layer_ids:
        QgsProject.instance().removeMapLayer(layer_id)

style_uri = '/Users/lfishgold/projects/raster-vision-examples/qgis/style.qml'
predict_dir = '/Users/lfishgold/raster-vision-data/examples/spacenet/vegas/buildings-local-output/predict/buildings-semantic_segmentation'
predict_paths = glob.glob(join(predict_dir, '*.tif'))
image_dir = '/Users/lfishgold/raster-vision-data/raw-data/spacenet-dataset/SpaceNet_Buildings_Dataset_Round2/spacenetV2_Train/AOI_2_Vegas/RGB-PanSharpen/'
gt_dir = '/Users/lfishgold/raster-vision-data/raw-data/spacenet-dataset/SpaceNet_Buildings_Dataset_Round2/spacenetV2_Train/AOI_2_Vegas/geojson/buildings/'

clear_layers()
max_scenes = min(5, len(predict_paths))
for s in range(max_scenes):
    predict_path = predict_paths[s]
    scene_id = splitext(basename(predict_path))[0]
    image_path = join(image_dir, 'RGB-PanSharpen_AOI_2_Vegas_img{}.tif'.format(scene_id))
    gt_path = join(gt_dir, 'buildings_AOI_2_Vegas_img{}.geojson'.format(scene_id))

    l = iface.addRasterLayer(image_path, scene_id + '-image')
    l = iface.addVectorLayer(gt_path, scene_id + '-gt', 'ogr')
    l = iface.addRasterLayer(predict_path, scene_id + '-predict')
    l.loadNamedStyle(style_uri)
iface.zoomToActiveLayer()