from os import environ
from os.path import join

data_path = '/opt/data/'
datasets_path = join(data_path, 'datasets')
results_path = join(data_path, 'results')
weights_path = join(data_path, 'weights')

s3_bucket = environ.get('S3_BUCKET')
s3_datasets_path = join('s3://{}'.format(s3_bucket), 'datasets')
s3_results_path = join('s3://{}'.format(s3_bucket), 'results')
s3_weights_path = join('s3://{}'.format(s3_bucket), 'weights')

TRAIN = 'train'
VALIDATION = 'validation'
TEST = 'test'
split_names = [TRAIN, VALIDATION, TEST]
