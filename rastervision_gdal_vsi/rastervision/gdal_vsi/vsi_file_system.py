from datetime import datetime
import os
from pathlib import Path
from typing import (List, Optional)
from urllib.parse import urlparse

from rastervision.pipeline.file_system import FileSystem

from osgeo import gdal


class VsiFileSystem(FileSystem):
    """A FileSystem to access files over any protocol supported by GDAL's VSI"""

    @staticmethod
    def uri_to_vsi_path(uri: str) -> str:
        """A function to convert Rasterio-like URIs to VSI path strings

        Args:
            uri: URI of the file, possibly nested within archives as follows
                 <archive_scheme>+<archive_URI>!path/to/contained/file.ext
                 Acceptable URI schemes are file, s3, gs, http, https, and ftp
                 Allowable archive schema are tar, zip, and gzip
        """
        parsed = urlparse(uri)
        scheme = parsed.scheme.split('+')[0]

        archive_content = uri.rfind('!')
        if archive_content == -1:
            # regular URI
            if scheme == 'http' or scheme == 'https' or scheme == 'ftp':
                return '/vsicurl/{}'.format(uri)
            elif scheme == 's3' or scheme == 'gs':
                return '/vsi{}/{}{}'.format(scheme, parsed.netloc, parsed.path)
            else:
                # assume file schema
                return os.path.abspath(
                    os.path.join(parsed.netloc, parsed.path))
        else:
            archive_target = uri.find('+')
            assert archive_target != -1

            if scheme in ['zip', 'tar', 'gzip']:
                return '/vsi{}/{}/{}'.format(
                    scheme,
                    VsiFileSystem.uri_to_vsi_path(
                        uri[archive_target + 1:archive_content]),
                    uri[archive_content + 1:])
            else:
                raise ValueError(
                    'Attempted access into archive with unsupported scheme "{}"'.
                    format(scheme))

    @staticmethod
    def matches_uri(vsipath: str, mode: str) -> bool:
        """Returns True if this FS can be used for the given URI/mode pair.

        Args:
            uri: URI of file
            mode: mode to open file in, 'r' or 'w'
        """
        if mode == 'r' and vsipath.startswith('/vsi'):
            return True
        elif mode == 'w' and vsipath.startswith(
                '/vsi') and '/vsicurl/' not in vsipath:
            return True
        else:
            return False

    @staticmethod
    def file_exists(vsipath: str, include_dir: bool = True) -> bool:
        """Check if a file exists.

        Args:
          uri: The URI to check
          include_dir: Include directories in check, if this file_system
            supports directory reads. Otherwise only return true if a single
            file exists at the URI.
        """
        file_stats = gdal.VSIStatL(vsipath)
        if include_dir:
            return True if file_stats else False
        else:
            return True if file_stats and not file_stats.IsDirectory(
            ) else False

    @staticmethod
    def read_bytes(vsipath: str) -> bytes:
        stats = gdal.VSIStatL(vsipath)
        if not stats or stats.IsDirectory():
            raise FileNotFoundError('{} does not exist'.format(vsipath))

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
        """Write string in data to URI."""
        VsiFileSystem.write_bytes(uri, data.encode())

    @staticmethod
    def sync_to_dir(src_dir: str, dst_dir_uri: str, delete: bool = False):
        """Syncs a local source directory to a destination directory.

        If the FileSystem is remote, this involves uploading.

        Args:
            src_dir: local source directory to sync from
            dst_dir_uri: A destination directory that can be synced to by this
                FileSystem
            delete: True if the destination should be deleted first.
        """

        def work(src, vsi_dest):
            gdal.Mkdir(vsi_dest, 0o777)

            for item in src.iterdir():
                item_vsi_dest = os.path.join(vsi_dest, item.name)
                if item.is_dir():
                    work(item, item_vsi_dest)
                else:
                    VsiFileSystem.copy_to(str(item), item_vsi_dest)

        stats = gdal.VSIStatL(dst_dir_uri)
        if stats:
            assert delete, 'Cannot overwrite existing files if delete=False'
            if stats.IsDirectory():
                gdal.RmdirRecursive(dst_dir_uri)
            else:
                gdal.Unlink(dst_dir_uri)

        src = Path(src_dir)
        assert src.exists() and src.is_dir(), \
            'Local source ({}) must be a directory'.format(src_dir)

        work(src, dst_dir_uri)

    @staticmethod
    def sync_from_dir(src_dir_uri: str, dst_dir: str, delete: bool = False):
        """Syncs a source directory to a local destination directory.

        If the FileSystem is remote, this involves downloading.

        Args:
            src_dir_uri: source directory that can be synced from by this FileSystem
            dst_dir: A local destination directory
            delete: True if the destination should be deleted first.
        """

        def work(vsi_src, dest):
            if dest.exists():
                assert dest.is_dir(
                ), 'Local target ({}) must be a directory'.format(dest)
            else:
                dest.mkdir()

            for item in gdal.ReadDir(vsi_src):
                item_vsi_src = os.path.join(vsi_src, item)
                target = dest.joinpath(item)
                if gdal.VSIStatL(item_vsi_src).IsDirectory():
                    work(item_vsi_src, target)
                else:
                    assert not target.exists() or delete, \
                        'Target location must not exist if delete=False'
                    VsiFileSystem.copy_from(item_vsi_src, str(target))

        stats = gdal.VSIStatL(src_dir_uri)
        assert stats and stats.IsDirectory(), 'Source must be a directory'

        work(src_dir_uri, Path(dst_dir))

    @staticmethod
    def copy_to(src_path: str, dst_uri: str):
        """Copy a local source file to a destination.

        If the FileSystem is remote, this involves uploading.

        Args:
            src_path: local path to source file
            dst_uri: uri of destination that can be copied to by this FileSystem
        """
        with open(src_path, 'rb') as f:
            buf = f.read()
        VsiFileSystem.write_bytes(dst_uri, buf)

    @staticmethod
    def copy_from(src_uri: str, dst_path: str):
        """Copy a source file to a local destination.

        If the FileSystem is remote, this involves downloading.

        Args:
            src_uri: uri of source that can be copied from by this FileSystem
            dst_path: local path to destination file
        """
        buf = VsiFileSystem.read_bytes(src_uri)
        with open(dst_path, 'wb') as f:
            f.write(buf)

    @staticmethod
    def local_path(vsipath: str, download_dir: str) -> str:
        """Return the path where a local copy should be stored.

        Args:
            uri: the URI of the file to be copied
            download_dir: path of the local directory in which files should
                be copied
        """
        filename = Path(vsipath).name
        return os.path.join(download_dir, filename)

    @staticmethod
    def last_modified(vsipath: str) -> Optional[datetime]:
        """Get the last modified date of a file.

        Args:
            uri: the URI of the file

        Returns:
            the last modified date in UTC of a file or None if this FileSystem
            does not support this operation.
        """
        stats = gdal.VSIStatL(vsipath)
        return datetime.fromtimestamp(stats.mtime) if stats else None

    @staticmethod
    def list_paths(vsipath: str, ext: Optional[str] = None) -> List[str]:
        """List paths rooted at URI.

        Optionally only includes paths with a certain file extension.

        Args:
            uri: the URI of a directory
            ext: the optional file extension to filter by
        """
        items = gdal.ReadDir(vsipath)
        ext = ext if ext else ''
        return [
            os.path.join(vsipath, item)  # This may not work for windows paths
            for item in filter(lambda x: x.endswith(ext), items)
        ]
