#!/usr/bin/env python3

import numpy as np
import rasterio as rio
import sys
import tempfile

from subprocess import Popen


# Type 'apt-get install python-gdal python3-gdal' before attempting to
# run this from inside of the Docker image.

vectors = None
v_file = sys.argv[1]
# mu_file = sys.argv[2]

for incoming in sys.argv[2:]:
    print(incoming)
    with rio.open(incoming) as dataset:
        data = dataset.read()
        shape = data.shape
        _vectors = data.reshape(shape[0], shape[1] * shape[2])
        if vectors is None:
            vectors = _vectors
        else:
            vectors = np.concatenate((vectors, _vectors), axis=1)

cov = np.cov(vectors)
w, v = np.linalg.eig(cov)
# mu = np.mean(vectors, axis=1)

print('w={}'.format(w))
print('v={}'.format(v))
# print('mu={}'.format(mu))

with open(v_file, 'wb') as outfile:
    np.save(outfile, v)

# with open(mu_file, 'wb') as outfile:
#     np.save(outfile, mu)
