import io
import os
import subprocess
from urllib.parse import urlparse

from rastervision.filesystem import (FileSystem, NotReadableError,
                                     NotWritableError)


class S3FileSystem(FileSystem):
    @staticmethod
    def matches_uri(uri: str, mode: str) -> bool:
        parsed_uri = urlparse(uri)
        return parsed_uri.scheme == 's3'

    @staticmethod
    def file_exists(uri: str) -> bool:
        # Lazily load boto
        import boto3
        import botocore

        s3 = boto3.resource('s3')
        parsed_uri = urlparse(uri)
        bucket = parsed_uri.netloc
        key = parsed_uri.path[1:]
        try:
            s3.Object(bucket, key).load()
        except botocore.exceptions.ClientError as e:
            return False
        return True

    @staticmethod
    def read_str(uri: str) -> str:
        return S3FileSystem.read_bytes(uri).decode('utf-8')

    @staticmethod
    def read_bytes(uri: str) -> bytes:
        # Lazily load boto
        import boto3
        import botocore

        parsed_uri = urlparse(uri)
        with io.BytesIO() as file_buffer:
            try:
                s3 = boto3.client('s3')
                s3.download_fileobj(parsed_uri.netloc, parsed_uri.path[1:],
                                    file_buffer)
                return file_buffer.getvalue()
            except botocore.exceptions.ClientError as e:
                raise NotReadableError('Could not read {}'.format(uri)) from e

    @staticmethod
    def write_str(uri: str, data: str) -> None:
        data = bytes(data, encoding='utf-8')
        S3FileSystem.write_bytes(uri, data)

    @staticmethod
    def write_bytes(uri: str, data: bytes) -> None:
        # Lazily load boto
        import boto3

        parsed_uri = urlparse(uri)
        bucket = parsed_uri.netloc
        key = parsed_uri.path[1:]
        with io.BytesIO(data) as str_buffer:
            try:
                s3 = boto3.client('s3')
                s3.upload_fileobj(str_buffer, bucket, key)
            except Exception as e:
                raise NotWritableError('Could not write {}'.format(uri)) from e

    @staticmethod
    def sync_from_dir(src_dir_uri: str,
                      dest_dir_uri: str,
                      delete: bool = False) -> None:
        command = ['aws', 's3', 'sync', src_dir_uri, dest_dir_uri]
        if delete:
            command.append('--delete')
        subprocess.run(command)

    @staticmethod
    def sync_to_dir(src_dir_uri: str, dest_dir_uri: str,
                    delete: bool = False) -> None:
        command = ['aws', 's3', 'sync', src_dir_uri, dest_dir_uri]
        if delete:
            command.append('--delete')
        subprocess.run(command)

    @staticmethod
    def copy_to(src_path: str, dst_uri: str) -> None:
        # Lazily load boto
        import boto3

        parsed_uri = urlparse(dst_uri)
        if os.path.isfile(src_path):
            try:
                s3 = boto3.client('s3')
                s3.upload_file(src_path, parsed_uri.netloc,
                               parsed_uri.path[1:])
            except Exception as e:
                raise NotWritableError(
                    'Could not write {}'.format(dst_uri)) from e
        else:
            S3FileSystem.sync_to_dir(src_path, dst_uri, delete=True)

    @staticmethod
    def copy_from(uri: str, path: str) -> None:
        # Lazily load boto
        import boto3
        import botocore

        parsed_uri = urlparse(uri)
        try:
            s3 = boto3.client('s3')
            s3.download_file(parsed_uri.netloc, parsed_uri.path[1:], path)
        except botocore.exceptions.ClientError:
            raise NotReadableError('Could not read {}'.format(uri))

    @staticmethod
    def local_path(uri: str, download_dir: str) -> None:
        parsed_uri = urlparse(uri)
        path = os.path.join(download_dir, 's3', parsed_uri.netloc,
                            parsed_uri.path[1:])
        return path
