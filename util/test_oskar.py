import logging
import unittest

from util.read_oskar_vis import *

from rascil.data_models.parameters import rascil_path

log = logging.getLogger(__name__)

class TestOskar(unittest.TestCase):

    @unittest.skip("Not compliant with data model")
    def test_visibility_from_oskar(self):
        for oskar_file in ["data/vis/vla_1src_6h/test_vla.vis",
                           "data/vis/vla_grid_6h/test_vla.vis"]:
            vis = import_visibility_from_oskar(rascil_path(oskar_file))
            self.assertEqual(len(numpy.unique(vis.antenna1))+1, len(vis.configuration.xyz))
            self.assertEqual(len(numpy.unique(vis.antenna2))+1, len(vis.configuration.xyz))

if __name__ == '__main__':
    unittest.main()
