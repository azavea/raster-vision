import unittest

from rastervision.utils.misc import (replace_nones_in_dict, set_nested_keys)


class TestMiscUtils(unittest.TestCase):
    def test_replace_nones_in_dict(self):
        d = {
            'one': None,
            'two': 2,
            'three': {
                'four': 4,
                'five': None,
                'six': [
                    {
                        'seven': None
                    },
                    {
                        'eight': None
                    },
                    {
                        'nine': 9
                    },
                ],
                'ten': {
                    'eleven': None
                }
            }
        }

        expected = {
            'one': 'A',
            'two': 2,
            'three': {
                'four': 4,
                'five': 'A',
                'six': [
                    {
                        'seven': 'A'
                    },
                    {
                        'eight': 'A'
                    },
                    {
                        'nine': 9
                    },
                ],
                'ten': {
                    'eleven': 'A'
                }
            }
        }

        self.assertEqual(replace_nones_in_dict(d, 'A'), expected)

    def test_set_nested_keys_finds_nested(self):
        d = {
            'one': 1,
            'two': 2,
            'three': {
                'four': 4,
                'five': 5,
                'six': [
                    {
                        'seven': 7
                    },
                    {
                        'eight': 8
                    },
                    {
                        'nine': 9
                    },
                ],
                'ten': {
                    'eleven': 11
                }
            }
        }

        expected = {
            'one': 1,
            'two': 2,
            'three': {
                'four': 4,
                'five': 55,
                'six': [
                    {
                        'seven': 7
                    },
                    {
                        'eight': 8
                    },
                    {
                        'nine': 9
                    },
                ],
                'ten': {
                    'eleven': 11
                }
            }
        }

        set_nested_keys(d, {'five': 55})

        self.assertEqual(d, expected)

    def test_set_nested_keys_ignores_missing_keys(self):
        d = {
            'one': 1,
            'two': 2,
            'three': {
                'four': 4,
                'five': 5,
                'six': [
                    {
                        'seven': 7
                    },
                    {
                        'eight': 8
                    },
                    {
                        'nine': 9
                    },
                ],
                'ten': {
                    'eleven': 11
                }
            }
        }

        expected = {
            'one': 1,
            'two': 2,
            'three': {
                'four': 4,
                'five': 5,
                'six': [
                    {
                        'seven': 7
                    },
                    {
                        'eight': 8
                    },
                    {
                        'nine': 9
                    },
                ],
                'ten': {
                    'eleven': 11
                }
            }
        }

        set_nested_keys(d, {'twenty': 20}, ignore_missing_keys=True)

        self.assertEqual(d, expected)

    def test_set_nested_keys_sets_missing_keys_in_dict(self):
        d = {
            'one': 1,
            'two': 2,
            'three': {
                'four': 4,
                'five': 5,
                'six': [
                    {
                        'seven': 7
                    },
                    {
                        'eight': 8
                    },
                    {
                        'nine': 9
                    },
                ],
                'ten': {
                    'eleven': 11
                }
            }
        }

        expected = {
            'one': 1,
            'two': 2,
            'three': {
                'four': 4,
                'five': 5,
                'six': [
                    {
                        'seven': 7
                    },
                    {
                        'eight': 8
                    },
                    {
                        'nine': 9
                    },
                ],
                'ten': {
                    'eleven': 11
                },
                'twelve': 12
            }
        }

        mod = {'three': {'twelve': 12}}

        set_nested_keys(d, mod, set_missing_keys=True)

        self.assertEqual(d, expected)
