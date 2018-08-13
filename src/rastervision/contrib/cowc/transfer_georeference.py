#!/usr/bin/env python

import os
import re
import sys

from subprocess import check_output, call

if not len(sys.argv) >= 4:
    print('Usage: {} <input_rgb.tif> <input_label.tif> <output_label.tif>'.
          format(sys.argv[0]))
    exit()

input_rgb = sys.argv[1]
input_label = sys.argv[2]
output_label = sys.argv[3]

# Get proj4 string
proj4 = check_output(['gdalsrsinfo', '-o', 'proj4', input_rgb], stderr=None)
proj4 = proj4[1:-2]

# Get upper left, lower right info
with open(os.devnull, 'w') as devnull:
    ullr = check_output(
        ['gdalinfo', input_rgb], stderr=devnull).decode('utf-8')
ul_re = re.compile(r'^Upper Left.*?([0-9\.]+).*?([0-9\.]+)', re.MULTILINE)
lr_re = re.compile(r'^Lower Right.*?([0-9\.]+).*?([0-9\.]+)', re.MULTILINE)
ul = re.search(ul_re, ullr)
lr = re.search(lr_re, ullr)

args = [
    'gdal_translate', '-a_srs', proj4, '-a_ullr',
    ul.group(1),
    ul.group(2),
    lr.group(1),
    lr.group(2), input_label, output_label
]

call(args)
