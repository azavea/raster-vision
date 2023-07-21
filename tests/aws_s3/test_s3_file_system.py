import unittest
from unittest.mock import patch

from botocore import UNSIGNED

from rastervision.aws_s3.s3_file_system import S3FileSystem


class TestS3FileSystem(unittest.TestCase):
    @patch.dict('os.environ', AWS_NO_SIGN_REQUEST='YES')
    def test_get_client_unsigned(self):
        s3 = S3FileSystem.get_client()
        self.assertEqual(s3._client_config.signature_version, UNSIGNED)


if __name__ == '__main__':
    unittest.main()
