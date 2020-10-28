""" Data path testing


"""

import unittest

from rascil.data_models.parameters import rascil_path, rascil_data_path
from rascil.processing_components.util.installation_checks import check_data_directory


class TestDataPath(unittest.TestCase):
    def test_rascil_data_path(self):
        data_path = rascil_data_path('configurations')
        check_data_directory()


if __name__ == '__main__':
    unittest.main()
