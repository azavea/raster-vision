import os

from rastervision.filesystems.filesystem import FileSystem


class LocalFileSystem(FileSystem):
    def matches_uri(uri: str) -> bool:
        return True

    def file_exists(uri: str) -> bool:
        return os.path.isfile(uri)

    def read(self, uri: str) -> bytearray:
        pass

    def write(self, uri: str, data: bytearray) -> int:
        pass
