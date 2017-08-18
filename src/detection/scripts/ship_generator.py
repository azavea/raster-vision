from os.path import join
from os import makedirs
from random import randrange
import argparse

import numpy as np
import pandas as pd
from PIL import Image
import rasterio
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.preprocessing import MinMaxScaler

from data_extent import data, folders
from rastervision.common.utils import _makedirs


def visualize(bbox_coords, windows, culled_windows, src_ds):
    x, y, x1, y1, x2, y2 = [], [], [], [], [], []
    full = []
    for r in windows:
        full.append([r[0], r[1], r[2], r[3]])
        full.append([r[0], r[1], r[3], r[2]])
    valid = []
    for r in culled_windows:
        valid.append([r[0], r[2], r[3], r[1]])
        valid.append([r[0], r[2], r[1], r[3]])
    for bbox in bbox_coords:
            x.append(bbox[2])
            y.append(bbox[0])
    for window in valid:
        x1.append(window[2])
        x1.append(window[3])
        y1.append(window[0])
        y1.append(window[1])
    for window in full:
        x2.append(window[2])
        x2.append(window[3])
        y2.append(window[0])
        y2.append(window[1])

    image = src_ds.read()
    image = np.transpose(image, [1, 2, 0])

    plt.imshow(image, origin='upper')
    plt.scatter(x, y, color='red')
    plt.scatter(x1, y1, color='blue')
    plt.scatter(x2, y2, color='green')
    plt.show()


def window_ordered_coords(bbox):
    # ymin, ymax, xmin, xmax
    return ((bbox[0], bbox[2]), (bbox[1], bbox[3]))


def bbox_ordered_coords(rect):
    # xmin, ymax, xmax, ymin
    return rect[0], rect[2], rect[1], rect[3]


def check_in_bbox(x, y, bbox):
    if (x > bbox[0] and x <= bbox[1]) and (y > bbox[2] and y <= bbox[3]):
        return True
    return False


def contains(big, sml):
    if big[0] <= sml[0] and big[1] <= sml[1] and big[2] >= sml[2] and \
       big[3] >= sml[3]:
        return True
    return False


def create_coordinate_rectangle(rectangle):
    rectangle_js = rectangle['geometry']['coordinates'][0]
    rect = [
        min(rectangle_js[0][0], rectangle_js[1][0]),  # minX/left
        max(rectangle_js[0][0], rectangle_js[1][0]),  # maxX/right
        min(rectangle_js[0][1], rectangle_js[2][1]),  # minY/bottom
        max(rectangle_js[0][1], rectangle_js[2][1])   # maxY/top
    ]
    print(rect)
    return [
        min(rectangle_js[0][0], rectangle_js[1][0]),  # minX/left
        max(rectangle_js[0][0], rectangle_js[1][0]),  # maxX/right
        min(rectangle_js[0][1], rectangle_js[2][1]),  # minY/bottom
        max(rectangle_js[0][1], rectangle_js[2][1])   # maxY/top
    ]


def rasterio_index(rect, src):
    ul = src.index(rect[0], rect[3])
    lr = src.index(rect[1], rect[2])

    minx = min(ul[0], lr[0])
    maxx = max(ul[0], lr[0])
    miny = min(ul[1], lr[1])
    maxy = max(ul[1], lr[1])

    if miny == maxy:
        maxy += 20
        print("I'm bad!", rect[2], rect[3], miny, maxy, minx, maxx)

    return [minx, maxx, miny, maxy]


def convert_to_pixels(src_ds, ogr_filename, img_corners):
    with open(ogr_filename) as ogr:
        ogr_dict = json.load(ogr)

    coord_rectangles = [create_coordinate_rectangle(feat)
                        for feat in ogr_dict['features'][::2]]
    pixel_coords = [rasterio_index(rect, src_ds) for rect in coord_rectangles]

    return pixel_coords


def expand_window_with_offset(bbox):
    """
    This commented out portion is another method of generating offset,
    with the new window location covering a larger area
    box_w = abs(bbox[0] - bbox[1])
    box_h = abs(bbox[2] - bbox[3])
    center_x = (bbox[0] + bbox[1])/2
    center_y = (bbox[2] + bbox[3])/2
    valid_w = 512 - box_w
    valid_h = 512 - box_h

    ul_X = randrange(center_x - valid_w + 100, center_x + valid_w - 255)
    ul_Y = randrange(center_y - valid_h + 100, center_y + valid_h - 255)
    """

    ul_X = randrange(bbox[0] - 100, bbox[0] + 100)
    ul_Y = randrange(bbox[2] - 100, bbox[2] + 100)

    return ul_X - 128, ul_X + 128, ul_Y - 128, ul_Y + 128


def expand_window_no_offset(bbox):
    if bbox[0] - 128 < 0 or bbox[0] + 128 < 0 or bbox[1] - 128 < 0 or \
       bbox[1] + 128 < 0:
        return 0, 256, 0, 256
    return [bbox[0] - 128, bbox[0] + 128, bbox[2] - 128, bbox[2] + 128]


def filter_windows(chip_boxes, ship_boxes, maxX, maxY):
    windows, ships = [], []
    for chip in chip_boxes:
        ships_in_chip = []
        valid_window = True
        if chip[0] < 0 or chip[1] < 0 or chip[2] > maxX or chip[3] > maxY:
            valid_window = False
        for ship in ship_boxes:
            # Check if ship overlaps with the window and if it does
            # check that the ship is completely in the window.
            if not rasterio.coords.disjoint_bounds(chip, ship):
                if not contains(chip, ship):
                    valid_window = False
                else:
                    # Convert ship pixel location to location inside chip
                    relative_ship = [ship[0] - chip[0], ship[2] - chip[0],
                                     ship[1] - chip[1], ship[3] - chip[1]]
                    ships_in_chip.append(relative_ship)
        if valid_window:
            windows.append(chip)
            ships.append(ships_in_chip)

    return windows, ships


def generate_chips(data_type, viz, datasets_path, singapore_path, img_dict,
                   active_inds, write_dir):
    chip_ships_list = []
    chip_ind = 0

    for img_ind in active_inds:
        dir_path = join(singapore_path, img_dict[img_ind]) + '/'
        img_name = data[img_ind][0]
        img_corners = data[img_ind][1]
        # src_path = dir_path + img_name                       # direct data
        src_path = dir_path + 'test_' + str(img_ind) + '.tif'  # warped data
        ogr_path = dir_path + 'ships_ogr_' + str(img_ind) + '.geojson'

        write_path = join(datasets_path, write_dir)
        _makedirs(write_path)

        with rasterio.open(src_path) as src_ds:
            flag = raw_input("Generate using " + img_name + "? ('q' to stop) ")
            if flag == 'q':
                break

            bbox_coords = convert_to_pixels(src_ds, ogr_path, img_corners)
            print(bbox_coords)
            ship_boxes = [rasterio.coords.BoundingBox(*bbox_ordered_coords(
                          bbox)) for bbox in bbox_coords]
            chip_coords = [expand_window_with_offset(bbox)
                           for bbox in bbox_coords]
            chip_boxes = [rasterio.coords.BoundingBox(*bbox_ordered_coords(
                          chip)) for chip in chip_coords]
            box_up = [expand_window_no_offset(bbox) for bbox in bbox_coords]
            windows, all_ships = filter_windows(chip_boxes, ship_boxes,
                                                src_ds.shape[0],
                                                src_ds.shape[1])

            masks = [window_ordered_coords(window) for window in windows]

            if viz:
                visualize(bbox_coords, box_up, windows, src_ds)

            for mask, ships_in_mask in zip(masks, all_ships):
                chip = str(chip_ind)
                if data_type == 'tif':
                    with rasterio.open(
                            write_path + chip + '.tif', 'w',
                            driver='GTiff', width=256, height=256, count=3,
                            dtype='uint16') as dst:
                        dst.write(src_ds.read(1, window=mask), indexes=1)  # r
                        dst.write(src_ds.read(2, window=mask), indexes=2)  # g
                        dst.write(src_ds.read(3, window=mask), indexes=3)  # b
                else:
                    with rasterio.open(
                            write_path + chip + '.jpg', 'w',
                            driver='JPEG', width=256, height=256, count=3,
                            dtype='uint8') as dst:
                        mask_img = src_ds.read([3, 2, 1], window=mask)
                        rescale_img = np.reshape(mask_img, (-1, 1))
                        scaler = MinMaxScaler(feature_range=(0, 255))
                        rescale_img = scaler.fit_transform(rescale_img)
                        mask_img_scaled = (np.reshape(rescale_img,
                                           mask_img.shape)).astype(np.uint8)
                        dst.write(mask_img_scaled)
                        dst.close()
                for ship in ships_in_mask:
                    write_debug_chip(chip, write_path, data_type, ships_in_mask
                                     )
                    chip_ships_list.append((chip + '.' + data_type, ship[0],
                                           ship[1], ship[2], ship[3], 'ship'))
                chip_ind += 1
                print(str(chip_ind) + " chip written")
            if flag == 'q':
                break
    # Once all chips have been created, write ship loc dataframe to csv
    csv_labels = ['filename', 'xmin', 'xmax', 'ymin', 'ymax', 'class_name']
    df = pd.DataFrame.from_records(chip_ships_list, columns=csv_labels)
    csv_write_path = write_path + 'ship_locations.csv'
    df.to_csv(csv_write_path, index=False)


def write_debug_chip(chip, write_path, data_type, ships):
    img_name = chip + '.' + data_type
    img_file_path = join(write_path, img_name)
    debug_path = join(write_path, 'debug')
    _makedirs(debug_path)
    img_write_path = join(debug_path, chip + '_debug.jpg')
    image = np.array(Image.open(img_file_path), dtype=np.uint8)

    # Create figure and axes
    fig, ax = plt.subplots(1)
    for ship in ships:
        rect = patches.Rectangle((ship[2], ship[0]), ship[3] - ship[2],
                                 ship[1] - ship[0],
                                 linewidth=1, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)

    plt.imshow(image, origin='upper')

    plt.savefig(img_write_path)
    plt.close()


def generate_all_chips(data_type, viz, datasets_path, singapore_path):
    img_dict = {}
    for img_ind, dir_name in enumerate(folders):
        img_dict[img_ind] = dir_name

    active_inds = range(0, 8)  # Train source indices
    write_dir = 'singapore_train/'
    generate_chips(data_type, viz, datasets_path, singapore_path, img_dict,
                   active_inds, write_dir)
    active_inds = [8, 9, 10]  # Validation source indices
    write_dir = 'singapore_val/'
    generate_chips(data_type, viz, datasets_path, singapore_path, img_dict,
                   active_inds, write_dir)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize')
    parser.add_argument('--output_type')

    return parser.parse_args()


def run():
    output = ['jpg', 'tif']
    viz = False
    args = parse_args()
    datasets_path = '/opt/Data/datasets/detection'
    singapore_path = join(datasets_path, 'singapore')

    if args.output_type in output:
        if visualize == 'true':
            viz = True
            print("""\nNote: In the visualization window, red marks ships, \
green marks windows centered
on ships prior to being offset and blue marks all valid offset windows that
will be used to generate chips from the provided image.\n""")
        generate_all_chips(args.output_type, viz, datasets_path,
                           singapore_path)
    else:
        "That's not a valid data type!"


if __name__ == '__main__':
    run()
