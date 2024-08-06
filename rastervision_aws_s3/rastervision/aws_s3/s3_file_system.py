from typing import Any, Iterator
import io
import os
import subprocess
from datetime import datetime
from urllib.parse import urlparse

import boto3
from tqdm.auto import tqdm

from rastervision.pipeline.file_system import (FileSystem, NotReadableError,
                                               NotWritableError)

AWS_S3 = 'aws_s3'


# Code from https://alexwlchan.net/2017/07/listing-s3-keys/
def get_matching_s3_objects(
        bucket: str,
        prefix: str = '',
        suffix: str = '',
        delimiter: str = '/',
        request_payer: str = 'None') -> Iterator[tuple[str, Any]]:
    """Generate objects in an S3 bucket.

    Args:
        bucket: Name of the S3 bucket.
        prefix: Only fetch objects whose key starts with this prefix.
        suffix: Only fetch objects whose keys end with this suffix.
    """
    s3 = S3FileSystem.get_client()
    kwargs = dict(
        Bucket=bucket,
        RequestPayer=request_payer,
        Delimiter=delimiter,
        Prefix=prefix,
    )
    while True:
        resp: dict = s3.list_objects_v2(**kwargs)
        dirs: list[dict[str, Any]] = resp.get('CommonPrefixes', {})
        files: list[dict[str, Any]] = resp.get('Contents', {})
        for obj in dirs:
            key: str = obj['Prefix']
            if key.startswith(prefix) and key.endswith(suffix):
                yield key, obj
        for obj in files:
            key: str = obj['Key']
            if key.startswith(prefix) and key.endswith(suffix):
                yield key, obj
        # The S3 API is paginated, returning up to 1000 keys at a time.
        # Pass the continuation token into the next response, until we
        # reach the final page (when this field is missing).
        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break


def get_matching_s3_keys(bucket: str,
                         prefix: str = '',
                         suffix: str = '',
                         delimiter: str = '/',
                         request_payer: str = 'None') -> Iterator[str]:
    """Generate the keys in an S3 bucket.

    Args:
        bucket: Name of the S3 bucket.
        prefix: Only fetch keys that start with this prefix.
        suffix: Only fetch keys that end with this suffix.
    """
    obj_iterator = get_matching_s3_objects(
        bucket,
        prefix=prefix,
        suffix=suffix,
        delimiter=delimiter,
        request_payer=request_payer)
    out = (key for key, _ in obj_iterator)
    return out


def progressbar(total_size: int, desc: str):
    return tqdm(
        total=total_size,
        desc=desc,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        mininterval=0.5,
        delay=5)


class S3FileSystem(FileSystem):
    """A FileSystem for interacting with files stored on AWS S3.

    Uses Everett configuration of form:
    ```
    [AWS_S3]
    requester_pays=True
    ```

    """

    @staticmethod
    def get_request_payer() -> str:
        # attempt to get from environ
        request_payer = os.getenv('AWS_REQUEST_PAYER', 'None')
        # attempt to get from RV config
        if request_payer == 'None':
            # Import here to avoid circular reference.
            from rastervision.pipeline import rv_config_ as rv_config
            requester_pays = rv_config.get_namespace_option(
                AWS_S3, 'requester_pays', as_bool=True)
            if requester_pays:
                request_payer = 'requester'
        return request_payer

    @staticmethod
    def get_session():
        return boto3.Session()

    @staticmethod
    def get_client():
        if os.getenv('AWS_NO_SIGN_REQUEST', '').lower() == 'yes':
            from botocore import UNSIGNED
            from botocore.config import Config
            s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
            return s3
        return S3FileSystem.get_session().client('s3')

    @staticmethod
    def matches_uri(uri: str, mode: str) -> bool:
        parsed_uri = urlparse(uri)
        return parsed_uri.scheme == 's3'

    @staticmethod
    def parse_uri(uri: str) -> tuple[str, str]:
        """Parse bucket name and key from an S3 URI."""
        parsed_uri = urlparse(uri)
        bucket, key = parsed_uri.netloc, parsed_uri.path[1:]
        return bucket, key

    @staticmethod
    def file_exists(uri: str, include_dir: bool = True) -> bool:
        # Lazily load boto
        import botocore

        parsed_uri = urlparse(uri)
        bucket = parsed_uri.netloc
        key = parsed_uri.path[1:]
        request_payer = S3FileSystem.get_request_payer()

        if include_dir:
            s3 = S3FileSystem.get_client()
            try:
                # Ensure key ends in slash so that this won't pick up files that
                # contain the key as a prefix, but aren't actually directories.
                # Example: if key is 'model' then we don't want to consider
                # model-123 a match.
                dir_key = key if key[-1] == '/' else key + '/'
                response = s3.list_objects_v2(
                    Bucket=bucket,
                    Prefix=dir_key,
                    MaxKeys=1,
                    RequestPayer=request_payer)
                if response['KeyCount'] == 0:
                    return S3FileSystem.file_exists(uri, include_dir=False)
                return True
            except botocore.exceptions.ClientError:
                return False
        else:
            s3r = S3FileSystem.get_session().resource('s3')
            try:
                s3r.Object(bucket, key).load(RequestPayer=request_payer)
                return True
            except botocore.exceptions.ClientError:
                return False

    @staticmethod
    def read_str(uri: str) -> str:
        return S3FileSystem.read_bytes(uri).decode('utf-8')

    @staticmethod
    def read_bytes(uri: str) -> bytes:
        import botocore

        s3 = S3FileSystem.get_client()
        request_payer = S3FileSystem.get_request_payer()
        bucket, key = S3FileSystem.parse_uri(uri)
        with io.BytesIO() as file_buffer:
            try:
                obj = s3.head_object(
                    Bucket=bucket, Key=key, RequestPayer=request_payer)
                file_size = obj['ContentLength']
                with progressbar(file_size, desc='Downloading') as bar:
                    s3.download_fileobj(
                        Bucket=bucket,
                        Key=key,
                        Fileobj=file_buffer,
                        Callback=lambda bytes: bar.update(bytes),
                        ExtraArgs={'RequestPayer': request_payer})
                return file_buffer.getvalue()
            except botocore.exceptions.ClientError as e:
                raise NotReadableError('Could not read {}'.format(uri)) from e

    @staticmethod
    def write_str(uri: str, data: str) -> None:
        data = bytes(data, encoding='utf-8')
        S3FileSystem.write_bytes(uri, data)

    @staticmethod
    def write_bytes(uri: str, data: bytes) -> None:
        s3 = S3FileSystem.get_client()
        bucket, key = S3FileSystem.parse_uri(uri)
        file_size = len(data)
        with io.BytesIO(data) as str_buffer:
            try:
                with progressbar(file_size, desc='Uploading') as bar:
                    s3.upload_fileobj(
                        Fileobj=str_buffer,
                        Bucket=bucket,
                        Key=key,
                        Callback=lambda bytes: bar.update(bytes))
            except Exception as e:
                raise NotWritableError(f'Could not write {uri}') from e

    @staticmethod
    def sync_from_dir(src_dir_uri: str, dst_dir: str,
                      delete: bool = False) -> None:  # pragma: no cover
        command = ['aws', 's3', 'sync', src_dir_uri, dst_dir]
        if delete:
            command.append('--delete')
        request_payer = S3FileSystem.get_request_payer()
        if request_payer:
            command.append('--request-payer')
        subprocess.run(command)

    @staticmethod
    def sync_to_dir(src_dir: str, dst_dir_uri: str,
                    delete: bool = False) -> None:  # pragma: no cover
        S3FileSystem.sync_from_dir(src_dir, dst_dir_uri, delete=delete)

    @staticmethod
    def copy_to(src_path: str, dst_uri: str) -> None:
        s3 = S3FileSystem.get_client()
        bucket, key = S3FileSystem.parse_uri(dst_uri)
        if os.path.isfile(src_path):
            file_size = os.path.getsize(src_path)
            try:
                with progressbar(file_size, desc='Uploading') as bar:
                    s3.upload_file(
                        Filename=src_path,
                        Bucket=bucket,
                        Key=key,
                        Callback=lambda bytes: bar.update(bytes))
            except Exception as e:
                raise NotWritableError(f'Could not write {dst_uri}') from e
        else:
            S3FileSystem.sync_to_dir(src_path, dst_uri, delete=True)

    @staticmethod
    def copy_from(src_uri: str, dst_path: str) -> None:
        import botocore

        s3 = S3FileSystem.get_client()
        request_payer = S3FileSystem.get_request_payer()
        bucket, key = S3FileSystem.parse_uri(src_uri)
        try:
            obj = s3.head_object(
                Bucket=bucket, Key=key, RequestPayer=request_payer)
            file_size = obj['ContentLength']
            with progressbar(file_size, desc=f'Downloading') as bar:
                s3.download_file(
                    Bucket=bucket,
                    Key=key,
                    Filename=dst_path,
                    Callback=lambda bytes: bar.update(bytes),
                    ExtraArgs={'RequestPayer': request_payer})
        except botocore.exceptions.ClientError:
            raise NotReadableError(f'Could not read {src_uri}')

    @staticmethod
    def local_path(uri: str, download_dir: str) -> None:
        parsed_uri = urlparse(uri)
        path = os.path.join(download_dir, 's3', parsed_uri.netloc,
                            parsed_uri.path[1:])
        return path

    @staticmethod
    def last_modified(uri: str) -> datetime:
        bucket, key = S3FileSystem.parse_uri(uri)
        s3 = S3FileSystem.get_client()
        request_payer = S3FileSystem.get_request_payer()
        head_data = s3.head_object(
            Bucket=bucket, Key=key, RequestPayer=request_payer)
        return head_data['LastModified']

    @staticmethod
    def list_paths(uri: str, ext: str = '', delimiter: str = '/') -> list[str]:
        request_payer = S3FileSystem.get_request_payer()
        if not uri.endswith('/'):
            uri += '/'
        parsed_uri = urlparse(uri)
        bucket = parsed_uri.netloc
        prefix = os.path.join(parsed_uri.path[1:])
        keys = get_matching_s3_keys(
            bucket,
            prefix,
            suffix=ext,
            delimiter=delimiter,
            request_payer=request_payer)
        paths = [os.path.join('s3://', bucket, key) for key in keys]
        return paths
