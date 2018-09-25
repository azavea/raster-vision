from rastervision.filesystem import FileSystem


class NoopFileSystem(FileSystem):
    @staticmethod
    def matches_uri(uri: str) -> bool:
        True

    @staticmethod
    def file_exists(uri: str) -> bool:
        False

    @staticmethod
    def read_str(uri: str) -> str:
        return ''

    @staticmethod
    def read_bytes(uri: str) -> bytes:
        return None

    @staticmethod
    def write_str(uri: str, data: str) -> None:
        pass

    @staticmethod
    def write_bytes(uri: str, data: bytes) -> None:
        pass

    @staticmethod
    def sync_from_dir(src_dir_uri: str,
                      dest_dir_uri: str,
                      delete: bool = False) -> None:
        pass

    @staticmethod
    def sync_to_dir(src_dir_uri: str, dest_dir_uri: str,
                    delete: bool = False) -> None:
        pass

    @staticmethod
    def copy_to(src_path: str, dst_uri: str) -> None:
        pass

    @staticmethod
    def copy_from(uri: str, path: str) -> None:
        pass

    @staticmethod
    def local_path(uri: str, download_dir: str) -> None:
        pass


def register_plugin(plugin_registry):
    plugin_registry.register_filesysystem(NoopFileSystem)
