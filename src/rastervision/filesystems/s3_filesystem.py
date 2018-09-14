import boto3

from rastervision.filesystems.filesystem import FileSystem
from urllib.parse import urlparse


class S3FileSystem(FileSystem):
    def matches_uri(uri: str) -> bool:
        parsed_uri = urlparse(uri)
        return parsed_uri.scheme == 's3'

    def file_exists(uri: str) -> bool:
        s3 = boto3.resource('s3')
        parsed_uri = urlparse(uri)
        bucket = s3.Bucket(parsed_uri.netloc)
        key = parsed_uri.path[1:]
        objs = list(bucket.objects.filter(Prefix=key))
        if len(objs) > 0 and objs[0].key == key:
            return True
        else:
            return False

    def read(self, uri: str) -> bytearray:
        pass

    def write(self, uri: str, data: bytearray) -> int:
        pass
