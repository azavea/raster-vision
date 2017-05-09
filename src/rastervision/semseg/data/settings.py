from os import environ
from os.path import join

data_path = '/opt/data/'
datasets_path = join(data_path, 'datasets/isprs')
results_path = join(data_path, 'results')
s3_bucket_name = environ.get('S3_BUCKET')
