#!/usr/bin/env python3

import numpy as np
import rasterio as rio
import sys
import tempfile

from subprocess import Popen


# Type 'apt-get install python-gdal python3-gdal' before attempting to
# run this from inside of the Docker image.

v_file = sys.argv[1]
# mu_file = sys.argv[2]
incoming = sys.argv[2]
outgoing = sys.argv[3]

with open(v_file, 'rb') as infile:
    v = np.load(infile)

# with open(mu_file, 'rb') as infile:
#     mu = np.load(infile)

args = ['gdal_calc.py']
args.extend(['-A', incoming, '--A_band=1'])
args.extend(['-B', incoming, '--A_band=2'])
args.extend(['-C', incoming, '--A_band=3'])
args.extend(['-D', incoming, '--A_band=4'])
args.extend(['--outfile=/tmp/red.tif'])
x = 0.5 if v[0,0] > 0 else -0.5
args.extend(['--calc=({})*A + ({})*B + ({})*C + ({})*D'.format(x*v[0,0], x*v[1,0], x*v[2,0], x*v[3,0])])
p = Popen(args)
p.wait()

args = ['gdal_calc.py']
args.extend(['-A', incoming, '--A_band=1'])
args.extend(['-B', incoming, '--A_band=2'])
args.extend(['-C', incoming, '--A_band=3'])
args.extend(['-D', incoming, '--A_band=4'])
args.extend(['--outfile=/tmp/green.tif'])
x = 2.0 if v[0,1] > 0 else -2.0
args.extend(['--calc=({})*A + ({})*B + ({})*C + ({})*D'.format(x*v[0,1], x*v[1,1], x*v[2,1], x*v[3,1])])
p = Popen(args)
p.wait()

args = ['gdal_calc.py']
args.extend(['-A', incoming, '--A_band=1'])
args.extend(['-B', incoming, '--A_band=2'])
args.extend(['-C', incoming, '--A_band=3'])
args.extend(['-D', incoming, '--A_band=4'])
args.extend(['--outfile=/tmp/blue.tif'])
x = 2.0 if v[0,2] > 0 else -2.0
args.extend(['--calc=({})*A + ({})*B + ({})*C + ({})*D'.format(x*v[0,2], x*v[1,2], x*v[2,2], x*v[3,2])])
p = Popen(args)
p.wait()

args = ['rm', '-f', outgoing]
p = Popen(args)
p.wait()

args = ['gdal_merge.py']
args.extend(['-separate', '-o', outgoing])
args.extend(['/tmp/red.tif', '/tmp/green.tif', '/tmp/blue.tif'])
p = Popen(args)
p.wait()
