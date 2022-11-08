from typing import Tuple
import io
import os
import subprocess
from datetime import datetime
from urllib.parse import urlparse

from everett.manager import ConfigurationMissingError
from tqdm.auto import tqdm

from rastervision.pipeline.file_system import (FileSystem, NotReadableError,
                                               NotWritableError)

AWS_S3 = 'aws_s3'


# Code from https://alexwlchan.net/2017/07/listing-s3-keys/
def get_matching_s3_objects(bucket, prefix='', suffix='',
                            request_payer='None'):
    """
    Generate objects in an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch objects whose key starts with
        this prefix (optional).
    :param suffix: Only fetch objects whose keys end with
        this suffix (optional).
    """
    import boto3
    s3 = boto3.client('s3')
    kwargs = {'Bucket': bucket, 'RequestPayer': request_payer}

    # If the prefix is a single string (not a tuple of strings), we can
    # do the filtering directly in the S3 API.
    if isinstance(prefix, str):
        kwargs['Prefix'] = prefix

    while True:

        # The S3 API response is a large blob of metadata.
        # 'Contents' contains information about the listed objects.
        resp = s3.list_objects_v2(**kwargs)

        try:
            contents = resp['Contents']
        except KeyError:
            return

        for obj in contents:
            key = obj['Key']
            if key.startswith(prefix) and key.endswith(suffix):
                yield obj

        # The S3 API is paginated, returning up to 1000 keys at a time.
        # Pass the continuation token into the next response, until we
        # reach the final page (when this field is missing).
        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break


def get_matching_s3_keys(bucket, prefix='', suffix='', request_payer='None'):
    """
    Generate the keys in an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (optional).
    :param suffix: Only fetch keys that end with this suffix (optional).
    """
    for obj in get_matching_s3_objects(bucket, prefix, suffix, request_payer):
        yield obj['Key']


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
    def get_request_payer():
        # Import here to avoid circular reference.
        from rastervision.pipeline import rv_config_ as rv_config
        try:
            s3_config = rv_config.get_namespace_config(AWS_S3)
            # 'None' needs the quotes because boto3 cannot handle None.
            return ('requester' if s3_config(
                'requester_pays', parser=bool, default='False') else 'None')
        except ConfigurationMissingError:
            return 'None'

    @staticmethod
    def get_session():
        # Lazily load boto
        import boto3
        return boto3.Session()

    @staticmethod
    def matches_uri(uri: str, mode: str) -> bool:
        parsed_uri = urlparse(uri)
        return parsed_uri.scheme == 's3'

    @staticmethod
    def parse_uri(uri: str) -> Tuple[str, str]:
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
            s3 = S3FileSystem.get_session().client('s3')
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

        s3 = S3FileSystem.get_session().client('s3')
        request_payer = S3FileSystem.get_request_payer()
        bucket, key = S3FileSystem.parse_uri(uri)
        with io.BytesIO() as file_buffer:
            try:
                file_size = s3.head_object(
                    Bucket=bucket, Key=key)['ContentLength']
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
        s3 = S3FileSystem.get_session().client('s3')
        bucket, key = S3FileSystem.parse_uri(uri)
        file_size = len(data)
        with io.BytesIO(data) as str_buffer:
            try:
                with progressbar(file_size, desc=f'Uploading') as bar:
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
        s3 = S3FileSystem.get_session().client('s3')
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

        s3 = S3FileSystem.get_session().client('s3')
        request_payer = S3FileSystem.get_request_payer()
        bucket, key = S3FileSystem.parse_uri(src_uri)
        try:
            file_size = s3.head_object(Bucket=bucket, Key=key)['ContentLength']
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
        s3 = S3FileSystem.get_session().client('s3')
        request_payer = S3FileSystem.get_request_payer()
        head_data = s3.head_object(
            Bucket=bucket, Key=key, RequestPayer=request_payer)
        return head_data['LastModified']

    @staticmethod
    def list_paths(uri, ext=''):
        request_payer = S3FileSystem.get_request_payer()
        parsed_uri = urlparse(uri)
        bucket = parsed_uri.netloc
        prefix = os.path.join(parsed_uri.path[1:])
        keys = get_matching_s3_keys(
            bucket, prefix, suffix=ext, request_payer=request_payer)
        return [os.path.join('s3://', bucket, key) for key in keys]
