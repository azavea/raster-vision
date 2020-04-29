import os
import stat
import time
from pathlib import Path

from rastervision2.pipeline.file_system import (FileSystem, NotReadableError, NotWritableError)

from osgeo import gdal
from rasterio.path import (Path, UnparsedPath, parse_path)

class VsiFileSystem(FileSystem):
    """A FileSystem to access files over any protocol supported by GDAL's VSI"""

    @staticmethod
    def uri_to_vsi_path(uri: str) -> str:
        parsed = urlparse(uri)
        scheme = parsed.scheme

        archive_content = uri.rfind('!')
        if archive_content == -1:
            # regular URI
            if scheme=='http' or scheme=='https' or scheme=='ftp':
                return '/vsicurl/{}'.format(uri)
            elif scheme=='s3' or scheme=='gs':
                return '/vsi{}/{}{}'.format(scheme, parsed.netloc, parsed.path)
            else:
                # assume file schema
                return os.path.abspath(os.path.join(parsed.netloc, parsed.path))
        else:
            archive_target = uri.find(':')
            assert archive_target != -1

            if scheme in ['zip','tar','gzip']:
                return '/vsi{}/{}/{}'.format(scheme, VsiFileSystem.uri_to_vsi_path(uri[archive_target+1:archive_content]), uri[archive_content+1:])
            else:
                raise ValueError('Attempted access into archive with unsupported scheme "{}"'.format(scheme))


    @staticmethod
    def matches_uri(uri: str, mode: str) -> bool:
        """Returns True if this FS can be used for the given URI/mode pair.

        Args:
            uri: URI of file
            mode: mode to open file in, 'r' or 'w'
        """
        parsed = parse_path(uri)
        if isinstance(parsed, UnparsedPath):
            return False
        else:
            vsi_path = VsiFileSystem.uri_to_vsi_path(uri)
            file_stats = gdal.VSIStatL(vsi_path)
            if mode == 'r':
                if file_stats:
                    return not file_stats.IsDirectory()
                else:
                    return False
            elif mode == 'w':
                return True # this may fail for vsicurl?
            else:
                raise ValueError('Unrecognized mode string: {}'.format(mode))


    @staticmethod
    @abstractmethod
    def file_exists(uri: str, include_dir: bool = True) -> bool:
        """Check if a file exists.

        Args:
          uri: The URI to check
          include_dir: Include directories in check, if this file_system
            supports directory reads. Otherwise only return true if a single
            file exists at the URI.
        """
        vsi_path = VsiFileSystem.uri_to_vsi_path(uri)
        file_stats = gdal.VSIStatL(vsi_path)
        if include_dir:
            True if file_stats else False
        else:
            return file_stats and not file_stats.IsDirectory()

    @staticmethod
    def read_str(uri: str) -> str:
        """Read contents of URI to a string."""
        read_bytes(uri).decode("UTF-8")

    @staticmethod
    def read_bytes_vsi(vsipath: str) -> bytes:
        try:
            handle = gdal.VSIFOpenL(vsipath, 'rb')
            stats = gdal.VSIStatL(vsipath)
            return gdal.VSIFReadL(1, stats.size, handle)
        finally:
            gdal.VSIFCloseL(handle)

    @staticmethod
    def read_bytes(uri: str) -> bytes:
        """Read contents of URI to bytes."""
        vsipath = VsiFileSystem.uri_to_vsi_path(uri)
        return VsiFileSystem.read_bytes_vsi(vsipath)

    @staticmethod
    def write_str(uri: str, data: str):
        """Write string in data to URI."""
        write_bytes(uri, data.encode())

    @staticmethod
    def write_bytes_vsi(vsipath: str, data: bytes):
        try:
            handle = gdal.VSIFOpenL(vsipath, 'wb')
            gdal.VSIFWriteL(data, 1, len(data), handle)
        finally:
            gdal.VSIFCloseL(handle)

    @staticmethod
    def write_bytes(uri: str, data: bytes):
        """Write bytes in data to URI."""
        vsipath = VsiFileSystem.uri_to_vsi_path(uri)
        VsiFileSystem.write_bytes_vsi(vsipath, data)

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
                    VsiFileSystem.copy_to_vsi(str(item), item_vsi_dest)

        vsipath = VsiFileSystem.uri_to_vi_path(dst_dir_uri)
        stats = gdal.VSIStatL(vsipath)
        if stats:
            assert delete, "Cannot overwrite existing files if delete=False"
            if stats.IsDirectory():
                gdal.RmdirRecursive(vsipath)
            else:
                gdal.Unlink(vsipath)

        src = Path(src_dir)
        assert src.exists() and src.is_dir(), "Local source ({}) must be a directory".format(src_dir)

        work(src, vsipath)

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
                assert dest.is_dir(), "Local target ({}) must be a directory".format(dest)
            else:
                dest.mkdir()

            for item in gdal.ReadDir(vsi_src):
                item_vsi_src = os.path.join(vsi_src, item)
                target = dest.joinpath(item)
                if gdal.VSIStatL(item_vsi_src).IsDirectory():
                    work(item_vsi_src, target)
                else:
                    assert not target.exists() or delete, "Target location may not exist if delete=False"
                    VsiFileSystem.copy_from_vsi(item_vsi_src, str(target))

        vsipath = VsiFileSystem.uri_to_vi_path(src_dir_uri)
        stats = gdal.VSIStatL(vsipath)
        assert stats and stats.IsDirectory(), "Source must be a directory"

        work(vsipath, Path(dst_dir))

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
        write_bytes(dst_uri, buf)

    @staticmethod
    def copy_to_vsi(src_path: str, dst_vsi: str):
        with open(src_path, 'rb') as f:
            buf = f.read()
        write_bytes_vsi(dst_vsi, buf)

    @staticmethod
    def copy_from(src_uri: str, dst_path: str):
        """Copy a source file to a local destination.

        If the FileSystem is remote, this involves downloading.

        Args:
            src_uri: uri of source that can be copied from by this FileSystem
            dst_path: local path to destination file
        """
        buf = read_bytes(src_uri)
        with open(dst_path, 'wb') as f:
            f.write(buf)

    @staticmethod
    def copy_from_vsi(src_vsi: str, dst_path: str):
        buf = read_bytes_vsi(src_vsi)
        with open(dst_path, 'wb') as f:
            f.write(buf)

    @staticmethod
    def local_path(uri: str, download_dir: str) -> str:
        """Return the path where a local copy should be stored.

        Args:
            uri: the URI of the file to be copied
            download_dir: path of the local directory in which files should
                be copied
        """
        vsipath = VsiFileSystem.uri_to_vsi_path(uri)
        filename = Path(vsipath).name
        return os.path.join(download_dir, filename)

    @staticmethod
    def last_modified(uri: str) -> Optional[datetime]:
        """Get the last modified date of a file.

        Args:
            uri: the URI of the file

        Returns:
            the last modified date in UTC of a file or None if this FileSystem
            does not support this operation.
        """
        vsipath = VsiFileSystem.uri_to_vsi_path(uri)
        stats = gdal.VSIStatL(vsipath)
        return time.gmtime(stats.mtime) if stats else None

    @staticmethod
    def list_paths(uri: str, ext: Optional[str] = None) -> List[str]:
        """List paths rooted at URI.

        Optionally only includes paths with a certain file extension.

        Args:
            uri: the URI of a directory
            ext: the optional file extension to filter by
        """
        vsipath = VsiFileSystem.uri_to_vsi_path(uri)
        items = gdal.ReadDir(vsipath)
        ext = ext if ext else ''
        return [os.path.join(vsipath, item) # This may not work for windows paths
                for item
                in filter(lambda x: x.endswith(ext), items)]
