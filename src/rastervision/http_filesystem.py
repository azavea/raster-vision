import os
import urllib
import urllib.request

from rastervision.filesystem import FileSystem
from typing import Union
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
                return  False
        except urllib.error.URLError:
            return False

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
