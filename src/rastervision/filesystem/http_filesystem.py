import os
import urllib
import urllib.request

from rastervision.filesystem.filesystem import FileSystem
from urllib.parse import urlparse


class HttpFileSystem(FileSystem):
    def matches_uri(uri: str) -> bool:
        parsed_uri = urlparse(uri)
        return parsed_uri.scheme in ['http', 'https']

    def file_exists(uri: str) -> bool:
        try:
            response = urllib.request.urlopen(uri)
            if response.getcode() == 200:
                return int(response.headers['content-length']) > 0
            else:
                return False
        except urllib.error.URLError:
            return False

    def read_str(uri: str) -> str:
        HttpFileSystem.read_bytes(uri).decode("utf8")

    def read_bytes(uri: str) -> bytes:
        with urllib.request.urlopen(uri) as req:
            return req.read()

    def write_str(uri: str, data: str) -> None:
        raise NotWritableError('Could not write {}'.format(uri))

    def write_bytes(uri: str, data: bytes) -> None:
        raise NotWritableError('Could not write {}'.format(uri))

    def sync_dir(src_dir_uri: str, dest_dir_uri: str, delete: bool=False) -> None:
        raise NotWritableError('Could not write {}'.format(dest_dir_uri))

    def copy_to(src_path: str, dst_uri: str) -> None:
        raise NotWritableError('Could not write {}'.format(dst_uri))

    def copy_from(uri: str, path: str) -> None:
        with urllib.request.urlopen(uri) as response:
            with open(path, 'wb') as out_file:
                try:
                    shutil.copyfileobj(response, out_file)
                except Exception:
                    raise NotReadableError('Could not read {}'.format(uri))

    def local_path(uri: str, download_dir: str) -> None:
        parsed_uri = urlparse(uri)
        path = os.path.join(download_dir, 'http', parsed_uri.netloc,
                            parsed_uri.path[1:])
        return path
