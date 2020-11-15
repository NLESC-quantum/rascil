""" Unit tests for visibility scatter gather and extend MS file


"""

import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility.gather_scatter import visibility_gather_time, visibility_gather_w, \
    visibility_scatter_time, visibility_scatter_w, visibility_scatter_channel, \
    visibility_gather_channel
from rascil.processing_components.visibility.iterators import vis_wslices, vis_timeslices
from rascil.processing_components.visibility.base import create_visibility, create_blockvisibility, extend_blockvisibility_to_ms
from rascil.data_models import rascil_path, rascil_data_path, BlockVisibility
from rascil.processing_components.visibility.base import create_blockvisibility_from_ms, create_visibility_from_ms, \
    export_blockvisibility_to_ms

import logging

log = logging.getLogger('logger')

log.setLevel(logging.WARNING)


class TestVisibilityGatherScatter(unittest.TestCase):

    def setUp(self):
        try:
            from casacore.tables import table  # pylint: disable=import-error
            self.casacore_available = True
            #            except ModuleNotFoundError:
        except:
            self.casacore_available = False

        # self.lowcore = create_named_configuration('LOWBD2-CORE')
        #
        # self.times = numpy.linspace(-300.0, 300.0, 11) * numpy.pi / 43200.0
        #
        # self.frequency = numpy.linspace(1e8, 1.5e9, 7)
        #
        # self.channel_bandwidth = numpy.array(7 * [self.frequency[1] - self.frequency[0]])
        #
        # self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')

    def test_vis_scatter_gather_timeslice_ms(self):
        # Reading
        msfile = rascil_data_path("vis/xcasa.ms")
        msoutfile = rascil_path("test_results/test_extend_xcasa.ms")
        # remove temp file if exists
        import os, shutil
        if os.path.exists(msoutfile):
            shutil.rmtree(msoutfile, ignore_errors=False)
        # open an existent file
        bvis_list = create_blockvisibility_from_ms(msfile)
        for bvis in bvis_list:
            if bvis is not None:
                vis_slices = vis_timeslices(bvis, 'auto')
                vis_list = visibility_scatter_time(bvis, vis_slices)
                for vis in vis_list:
                    extend_blockvisibility_to_ms(msoutfile, vis)


if __name__ == '__main__':
    unittest.main()
