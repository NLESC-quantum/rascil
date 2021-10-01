""" Unit processing_components for rascil advise

"""
import logging
import unittest
import shutil

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.apps.rascil_advise import cli_parser, advise
from rascil.data_models.parameters import rascil_path
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import (
    export_blockvisibility_to_ms,
    concatenate_blockvisibility_frequency,
)
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.simulation import ingest_unittest_visibility

log = logging.getLogger("rascil-logger")
log.setLevel(logging.WARNING)


class TestRASCILAdvise(unittest.TestCase):
    def make_MS(self, nfreqwin=1, dopol=False, zerow=False):

        self.low = create_named_configuration("LOWBD2", rmax=750.0)
        self.freqwin = nfreqwin
        self.ntimes = 3
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)

        if self.freqwin > 1:
            self.channelwidth = numpy.array(
                self.freqwin * [self.frequency[1] - self.frequency[0]]
            )
        else:
            self.channelwidth = numpy.array([1e6])

        if dopol:
            self.vis_pol = PolarisationFrame("linear")
            self.image_pol = PolarisationFrame("stokesIQUV")
            f = numpy.array([100.0, 20.0, 0.0, 0.0])
        else:
            self.vis_pol = PolarisationFrame("stokesI")
            self.image_pol = PolarisationFrame("stokesI")
            f = numpy.array([100.0])

        self.phasecentre = SkyCoord(
            ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.bvis_list = [
            ingest_unittest_visibility(
                self.low,
                [self.frequency[i]],
                [self.channelwidth[i]],
                self.times,
                self.vis_pol,
                self.phasecentre,
                zerow=zerow,
            )
            for i in range(nfreqwin)
        ]

        shutil.rmtree(
            rascil_path("test_results/test_rascil_advise.ms"), ignore_errors=True
        )
        if nfreqwin > 1:
            self.bvis_list = [concatenate_blockvisibility_frequency(self.bvis_list)]

        export_blockvisibility_to_ms(
            rascil_path("test_results/test_rascil_advise.ms"), self.bvis_list
        )

    def setUp(self) -> None:

        self.results_dir = rascil_path("test_results")

        parser = cli_parser()
        self.args = parser.parse_args([])
        self.args.ingest_msname = rascil_path("test_results/test_rascil_advise.ms")

    def tearDown(self) -> None:
        shutil.rmtree(
            rascil_path("test_results/test_rascil_advise.ms"), ignore_errors=True
        )

    def test_advise(self):

        self.make_MS()

        advice = advise(self.args)

        numpy.testing.assert_almost_equal(advice["cellsize"], 0.0010626331874509759)
        numpy.testing.assert_almost_equal(
            advice["freq_sampling_primary_beam"], 2381946.1861413997
        )
        numpy.testing.assert_almost_equal(advice["nwpixels"], 18)
        numpy.testing.assert_almost_equal(advice["nwpixels_image"], 494)
        numpy.testing.assert_almost_equal(
            advice["time_sampling_primary_beam"], 2572.5018810327115
        )


if __name__ == "__main__":
    unittest.main()
