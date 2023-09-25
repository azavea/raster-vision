from os.path import join
import unittest

from rastervision.pipeline.file_system import (get_tmp_dir, str_to_file,
                                               LocalFileSystem)
from rastervision.gdal_vsi.vsi_file_system import VsiFileSystem

fs = VsiFileSystem


class TestVsiFileSystem(unittest.TestCase):
    def test_uri_to_vsi_path(self):
        self.assertEqual(fs.uri_to_vsi_path('/a/b/c'), '/a/b/c')
        self.assertEqual(fs.uri_to_vsi_path('http://a/b/c'), '/vsicurl/a/b/c')
        self.assertEqual(fs.uri_to_vsi_path('https://a/b/c'), '/vsicurl/a/b/c')
        self.assertEqual(fs.uri_to_vsi_path('ftp://a/b/c'), '/vsicurl/a/b/c')
        self.assertEqual(fs.uri_to_vsi_path('s3://a/b/c'), '/vsis3/a/b/c')
        self.assertEqual(fs.uri_to_vsi_path('gs://a/b/c'), '/vsigs/a/b/c')

    def test_uri_to_vsi_path_archive(self):
        with self.assertRaises(ValueError):
            _ = fs.uri_to_vsi_path('wrongscheme+s3://a/b!c')

        self.assertEqual(
            fs.uri_to_vsi_path('zip+s3://a/b!c'), '/vsizip/vsis3/a/b/c')
        self.assertEqual(
            fs.uri_to_vsi_path('gzip+s3://a/b!c'), '/vsigzip/vsis3/a/b/c')
        self.assertEqual(
            fs.uri_to_vsi_path('tar+s3://a/b!c'), '/vsitar/vsis3/a/b/c')

    def test_matches_uri(self):
        self.assertFalse(fs.matches_uri('/a/b/c', 'r'))
        self.assertTrue(fs.matches_uri('/vsis3/a/b/c', 'r'))
        self.assertTrue(fs.matches_uri('/vsis3/a/b/c', 'w'))
        self.assertTrue(fs.matches_uri('/vsicurl/a/b/c', 'r'))
        self.assertFalse(fs.matches_uri('/vsicurl/a/b/c', 'w'))

    def test_local_path(self):
        vsipath = '/vsicurl/a/b/c'
        self.assertEqual(fs.local_path(vsipath, '/'), '/c')

    def test_read_write_bytes(self):
        with get_tmp_dir() as tmp_dir:
            path = join(tmp_dir, 'test.bin')
            path_vsi = fs.uri_to_vsi_path(path)
            bytes_in = bytes([0x00, 0x01, 0x02])
            fs.write_bytes(path_vsi, bytes_in)
            bytes_out = fs.read_bytes(path_vsi)
            self.assertEqual(bytes_in, bytes_out)

        with self.assertRaises(FileNotFoundError):
            fs.read_bytes(path_vsi)

    def test_read_write_str(self):
        with get_tmp_dir() as tmp_dir:
            path = join(tmp_dir, 'test.txt')
            path_vsi = fs.uri_to_vsi_path(path)
            str_in = 'abc'
            fs.write_str(path_vsi, str_in)
            str_out = fs.read_str(path_vsi)
            self.assertEqual(str_in, str_out)

    def test_list_paths(self):
        with get_tmp_dir() as tmp_dir:
            dir_vsi = fs.uri_to_vsi_path(tmp_dir)
            str_to_file('abc', join(tmp_dir, '1.txt'))
            str_to_file('def', join(tmp_dir, '2.txt'))
            str_to_file('ghi', join(tmp_dir, '3.tiff'))
            paths = fs.list_paths(dir_vsi, ext='txt')
            self.assertSetEqual(
                set(paths),
                set([join(tmp_dir, '1.txt'),
                     join(tmp_dir, '2.txt')]))

    def test_sync_to_from(self):
        with get_tmp_dir() as src, get_tmp_dir() as dst:
            src_vsi = fs.uri_to_vsi_path(src)
            dst_vsi = fs.uri_to_vsi_path(dst)
            str_to_file('abc', join(src, '1.txt'))
            str_to_file('def', join(src, '2.txt'))
            str_to_file('ghi', join(src, 'subdir', '3.txt'))
            fs.sync_to_dir(src_vsi, dst_vsi, delete=True)
            paths = fs.list_paths(dst_vsi)
            self.assertSetEqual(
                set(paths),
                set([
                    join(dst, 'subdir'),
                    join(dst, '1.txt'),
                    join(dst, '2.txt'),
                ]))
            paths = fs.list_paths(dst_vsi, ext='txt')
            self.assertSetEqual(
                set(paths), set([
                    join(dst, '1.txt'),
                    join(dst, '2.txt'),
                ]))
            paths = fs.list_paths(join(dst_vsi, 'subdir'))
            self.assertSetEqual(
                set(paths), set([join(dst, 'subdir', '3.txt')]))

            with self.assertRaises(FileExistsError):
                fs.sync_to_dir(src_vsi, dst_vsi, delete=False)

            with self.assertRaises(ValueError):
                fs.sync_to_dir(join(src, '1.txt'), dst_vsi, delete=True)

            fs.sync_from_dir(src_vsi, dst_vsi, delete=True)
            paths = fs.list_paths(src_vsi)
            self.assertSetEqual(
                set(paths),
                set([
                    join(src, 'subdir'),
                    join(src, '1.txt'),
                    join(src, '2.txt'),
                ]))
            paths = fs.list_paths(src_vsi, ext='txt')
            self.assertSetEqual(
                set(paths), set([
                    join(src, '1.txt'),
                    join(src, '2.txt'),
                ]))
            paths = fs.list_paths(join(src, 'subdir'))
            self.assertSetEqual(
                set(paths), set([join(src, 'subdir', '3.txt')]))

            with self.assertRaises(FileExistsError):
                fs.sync_from_dir(src_vsi, dst_vsi, delete=False)

            with self.assertRaises(ValueError):
                fs.sync_from_dir(src_vsi, join(dst, '1.txt'), delete=True)

            with self.assertRaises(ValueError):
                fs.sync_from_dir(join(src, '1.txt'), dst_vsi, delete=True)

    def test_last_modified(self):
        with get_tmp_dir() as tmp_dir:
            path = join(tmp_dir, '1.txt')
            str_to_file('abc', path)
            path_vsi = fs.uri_to_vsi_path(path)
            self.assertEqual(
                fs.last_modified(path_vsi).timestamp(),
                int(LocalFileSystem.last_modified(path).timestamp()))

    def test_file_exists(self):
        with get_tmp_dir() as tmp_dir:
            path = join(tmp_dir, '1.txt')
            str_to_file('abc', path)
            path_vsi = fs.uri_to_vsi_path(path)
            self.assertTrue(fs.file_exists(path_vsi, include_dir=False))
            dir_vsi = fs.uri_to_vsi_path(tmp_dir)
            self.assertTrue(fs.file_exists(dir_vsi))


if __name__ == '__main__':
    unittest.main()
