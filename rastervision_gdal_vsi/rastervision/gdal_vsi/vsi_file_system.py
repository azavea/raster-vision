import os
from os.path import join
from pathlib import Path
import re
from datetime import datetime
from urllib.parse import urlparse

from rastervision.pipeline.file_system import FileSystem

from osgeo import gdal

ARCHIVE_URI_FORMAT = (
    r'^(?P<archive_scheme>[^+]+)\+(?P<archive_uri>[^!]+)!(?P<file_path>.+)$')
URI_SCHEME_TO_VSI = {
    'http': 'vsicurl',
    'https': 'vsicurl',
    'ftp': 'vsicurl',
    's3': 'vsis3',
    'gs': 'vsigs',
}
ARCHIVE_SCHEME_TO_VSI = {
    'zip': 'vsizip',
    'gzip': 'vsigzip',
    'tar': 'vsitar',
}


class VsiFileSystem(FileSystem):
    """A FileSystem to access files over any protocol supported by GDAL's VSI"""

    @staticmethod
    def uri_to_vsi_path(uri: str) -> str:
        """A function to convert Rasterio-like URIs to VSI path strings

        Args:
            uri: URI of the file, possibly nested within archives as follows
                <archive_scheme>+<archive_URI>!path/to/contained/file.ext.
                Acceptable URI schemes are file, s3, gs, http, https, and ftp.
                Allowable archive schema are tar, zip, and gzip.

        Raises:
            ValueError: If URI format or schema is invalid.
        """
        parsed = VsiFileSystem.parse_archive_format(uri)
        if parsed is None:
            # regular URI
            parsed = urlparse(uri)
            scheme, netloc, path = parsed.scheme, parsed.netloc, parsed.path
            if scheme in URI_SCHEME_TO_VSI:
                return join('/', URI_SCHEME_TO_VSI[scheme], f'{netloc}{path}')
            # assume file schema
            return os.path.abspath(join(netloc, path))

        archive_scheme = parsed['archive_scheme']
        archive_uri = parsed['archive_uri']
        file_path = parsed['file_path']
        try:
            vsi_archive_scheme = ARCHIVE_SCHEME_TO_VSI[archive_scheme]
        except KeyError:
            raise ValueError('Expected archive scheme to be one of "zip", '
                             f'"tar", or "gzip". Found "{archive_scheme}".')
        vsi_archive_uri = VsiFileSystem.uri_to_vsi_path(archive_uri)
        vsipath = join(f'/{vsi_archive_scheme}{vsi_archive_uri}', file_path)
        return vsipath

    @staticmethod
    def parse_archive_format(uri: str) -> re.Match:
        match = re.match(ARCHIVE_URI_FORMAT, uri)
        if match is None:
            return None
        return match.groupdict()

    @staticmethod
    def matches_uri(uri: str, mode: str) -> bool:
        if not uri.startswith('/vsi'):
            return False
        if mode == 'w' and '/vsicurl/' in uri:
            return False
        return True

    @staticmethod
    def file_exists(vsipath: str, include_dir: bool = True) -> bool:
        file_stats = gdal.VSIStatL(vsipath)
        if include_dir:
            return bool(file_stats)
        else:
            return file_stats and not file_stats.IsDirectory()

    @staticmethod
    def read_bytes(vsipath: str) -> bytes:
        stats = gdal.VSIStatL(vsipath)
        if not stats or stats.IsDirectory():
            raise FileNotFoundError(f'{vsipath} does not exist')

        try:
            retval = bytes()
            handle = gdal.VSIFOpenL(vsipath, 'rb')
            bytes_left = stats.size
            while bytes_left > 0:
                bytes_to_read = min(bytes_left, 1 << 30)
                retval += gdal.VSIFReadL(1, bytes_to_read, handle)
                bytes_left -= bytes_to_read
            return retval
        finally:
            gdal.VSIFCloseL(handle)

    @staticmethod
    def read_str(uri: str) -> str:
        """Read contents of URI to a string."""
        return VsiFileSystem.read_bytes(uri).decode('UTF-8')

    @staticmethod
    def write_bytes(vsipath: str, data: bytes):
        try:
            handle = gdal.VSIFOpenL(vsipath, 'wb')
            gdal.VSIFWriteL(data, 1, len(data), handle)
        finally:
            gdal.VSIFCloseL(handle)

    @staticmethod
    def write_str(uri: str, data: str):
        VsiFileSystem.write_bytes(uri, data.encode())

    @staticmethod
    def sync_to_dir(src_dir: str, dst_dir_uri: str, delete: bool = False):
        def work(src: Path, vsi_dest: str):
            gdal.Mkdir(vsi_dest, 0o777)

            for item in src.iterdir():
                item_vsi_dest = join(vsi_dest, item.name)
                if item.is_dir():
                    work(item, item_vsi_dest)
                else:
                    VsiFileSystem.copy_to(str(item), item_vsi_dest)

        stats = gdal.VSIStatL(dst_dir_uri)
        if stats:
            if not delete:
                raise FileExistsError(
                    'Target location must not exist if delete=False')
            if stats.IsDirectory():
                gdal.RmdirRecursive(dst_dir_uri)
            else:
                gdal.Unlink(dst_dir_uri)

        src = Path(src_dir)
        if not (src.exists() and src.is_dir()):
            raise ValueError('Source must be a directory')

        work(src, dst_dir_uri)

    @staticmethod
    def sync_from_dir(src_dir_uri: str, dst_dir: str, delete: bool = False):
        def work(vsi_src: str, dest: Path):
            if dest.exists():
                if not dest.is_dir():
                    raise ValueError(
                        f'Local target ({dest}) must be a directory')
            else:
                dest.mkdir()

            for item in VsiFileSystem.list_children(vsi_src):
                item_vsi_src = join(vsi_src, item)
                target = dest.joinpath(item)
                if gdal.VSIStatL(item_vsi_src).IsDirectory():
                    work(item_vsi_src, target)
                else:
                    if target.exists() and not delete:
                        raise FileExistsError(
                            'Target location must not exist if delete=False')
                    VsiFileSystem.copy_from(item_vsi_src, str(target))

        stats = gdal.VSIStatL(src_dir_uri)
        if not (stats and stats.IsDirectory()):
            raise ValueError('Source must be a directory')

        work(src_dir_uri, Path(dst_dir))

    @staticmethod
    def copy_to(src_path: str, dst_uri: str):
        with open(src_path, 'rb') as f:
            buf = f.read()
        VsiFileSystem.write_bytes(dst_uri, buf)

    @staticmethod
    def copy_from(src_uri: str, dst_path: str):
        buf = VsiFileSystem.read_bytes(src_uri)
        with open(dst_path, 'wb') as f:
            f.write(buf)

    @staticmethod
    def local_path(vsipath: str, download_dir: str) -> str:
        filename = Path(vsipath).name
        return join(download_dir, filename)

    @staticmethod
    def last_modified(vsipath: str) -> datetime | None:
        stats = gdal.VSIStatL(vsipath)
        return datetime.fromtimestamp(stats.mtime) if stats else None

    @staticmethod
    def list_paths(vsipath: str, ext: str | None = None) -> list[str]:
        items = VsiFileSystem.list_children(vsipath, ext=ext)
        paths = [join(vsipath, item) for item in items]
        return paths

    @staticmethod
    def list_children(vsipath: str, ext: str | None = None) -> list[str]:
        """List filenames of children rooted at URI.

        Optionally only includes filenames with a certain file extension.

        Args:
            uri: The URI of a directory.
            ext: The optional file extension to filter by.

        Returns:
            List of filenames excluding "." or "..".
        """
        ext = ext if ext else ''
        items = gdal.ReadDir(vsipath)
        items = [item for item in items if item not in ['.', '..']]
        items = [item for item in items if item.endswith(ext)]
        return items
