import subprocess

deps = [
    'rastervision', 'gdal', 'shapely', 'geopandas', 'numpy', 'PIL', 'pyproj',
    'sklearn', 'scipy', 'scipy', 'cv2', 'imageio', 'tensorboard',
    'albumentations', 'cython', 'pycocotools', 'matplotlib', 'numpy'
]

other_deps = [
    'pygeos',
    'pystac',
    'joblib',
    'threadpoolctl',
    'tqdm',
    'boto3',
    'future',
    'psutil',
    'triangle',
    'click',
    'pydantic',
    'typing_extensions',
    'everett',
    'six',
]

seg_fault_pairs = []
good_pairs = []
'''
for d1 in deps:
    for d2 in deps:
        cmd = ['python', '_test_import_order.py', d1, d2]
        print(cmd)
        result = subprocess.Popen(cmd)
        output = result.communicate()[0]
        return_code = result.returncode
        if return_code != 0:
            seg_fault_pairs.append((d1, d2))
        else:
            good_pairs.append((d1, d2))
'''

for d1 in other_deps:
    cmd = ['python', '_test_import_order.py', d1, d1]
    print(cmd)
    result = subprocess.Popen(cmd)
    output = result.communicate()[0]
    return_code = result.returncode
    if return_code != 0:
        seg_fault_pairs.append((d1, d1))
    else:
        good_pairs.append((d1, d1))

print('good: ', good_pairs)
print('segfaulted: ', seg_fault_pairs)
