import io
import os
import boto3
import botocore

from rastervision.filesystem.filesystem import *
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

    def read_str(uri: str) -> str:
        return S3FileSystem.read_bytes(uri).decode('utf-8')

    def read_bytes(uri: str) -> bytes:
        parsed_uri = urlparse(uri)
        with io.BytesIO() as file_buffer:
            try:
                s3 = boto3.client('s3')
                s3.download_fileobj(parsed_uri.netloc, parsed_uri.path[1:],
                                    file_buffer)
                return file_buffer.getvalue()
            except botocore.exceptions.ClientError as e:
                raise NotReadableError('Could not read {}'.format(uri)) from e

    def write_str(uri: str, data: str) -> None:
        data = bytes(data, encoding='utf-8')
        S3FileSystem.write_bytes(uri, data)

    def write_bytes(uri: str, data: bytes) -> None:
        parsed_uri = urlparse(uri)
        bucket = parsed_uri.netloc
        key = parsed_uri.path[1:]
        with io.BytesIO(data) as str_buffer:
            try:
                s3 = boto3.client('s3')
                s3.upload_fileobj(str_buffer, bucket, key)
            except Exception as e:
                raise NotWritableError('Could not write {}'.format(uri)) from e

    def sync_dir(src_dir_uri: str, dest_dir_uri: str, delete: bool=False) -> None:
        command = ['aws', 's3', 'sync', src_dir_uri, dest_dir_uri]
        if delete:
            command.append('--delete')
        subprocess.run(command)

    def copy_to(src_path: str, dst_uri: str) -> None:
        parsed_uri = urlparse(dst_uri)
        if os.path.isfile(src_path):
            try:
                s3 = boto3.client('s3')
                s3.upload_file(src_path, parsed_uri.netloc,
                               parsed_uri.path[1:])
            except Exception as e:
                raise NotWritableError('Could not write {}'.format(dst_uri)) from e
        else:
            sync_dir(src_path, dst_uri, delete=True)

    def copy_from(uri: str, path: str) -> None:
        parsed_uri = urlparse(uri)
        try:
            s3 = boto3.client('s3')
            s3.download_file(parsed_uri.netloc, parsed_uri.path[1:], path)
        except botocore.exceptions.ClientError:
            raise NotReadableError('Could not read {}'.format(uri))

    def local_path(uri: str, download_dir: str) -> None:
        parsed_uri = urlparse(uri)
        path = os.path.join(download_dir, 's3', parsed_uri.netloc,
                            parsed_uri.path[1:])
        return path
