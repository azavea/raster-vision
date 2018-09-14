import urllib
import urllib.request

from rastervision.filesystems.filesystem import FileSystem
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

    def read(self, uri: str) -> bytearray:
        pass

    def write(self, uri: str, data: bytearray) -> int:
        pass
