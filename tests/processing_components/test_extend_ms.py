""" Unit tests for visibility scatter gather and extend MS file


"""

import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility.base import create_blockvisibility, extend_blockvisibility_to_ms
from rascil.data_models import rascil_path, rascil_data_path, BlockVisibility
from rascil.processing_components.visibility.base import create_blockvisibility_from_ms, export_blockvisibility_to_ms

import logging

log = logging.getLogger('logger')

log.setLevel(logging.WARNING)


class TestExtendMS(unittest.TestCase):

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

    def test_extend_ms(self):
        # Reading
        msfile = rascil_data_path("vis/xcasa.ms")
        msoutfile = rascil_path("test_results/test_extend_xcasa.ms")
        # remove temp file if exists
        import os, shutil
        if os.path.exists(msoutfile):
            shutil.rmtree(msoutfile, ignore_errors=False)
        # open an existent file
        bvis = create_blockvisibility_from_ms(msfile)[0]
        bvis_list = [bv[1] for bv in bvis.groupby("time", squeeze=False)]
        for bvis in bvis_list:
            extend_blockvisibility_to_ms(msoutfile, bvis)


if __name__ == '__main__':
    unittest.main()
