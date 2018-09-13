from rastervision.filesystem import FileSystem
from typing import Union


class LocalFileSystem(FileSystem):

    def matches_uri(uri: str) -> bool:
        return True

    def file_exists(uri: str) -> bool:
        return True

    def open(uri: str) -> Union[FileSystem, None]:
        return None

    def close(self) -> None:
        pass

    def read(self) -> bytearray:
        pass

    def write(self, data: bytearray) -> int:
        pass

    def seek(self, offset: int, whence: int) -> int:
        pass

    def tell(self) -> int:
        pass
