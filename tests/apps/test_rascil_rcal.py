""" Unit processing_components for rascil advise

"""
import sys
import os
import logging
import unittest
import shutil

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.apps.rascil_rcal import cli_parser, rcal_simulator
from rascil.data_models import rascil_path, Skycomponent, import_gaintable_from_hdf5
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import (
    export_blockvisibility_to_ms,
    dft_skycomponent_visibility,
    create_gaintable_from_blockvisibility,
    simulate_gaintable,
    apply_gaintable,
    qa_visibility,
)
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.simulation import ingest_unittest_visibility

log = logging.getLogger("rascil-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


class TestRASCILRCAL(unittest.TestCase):
    def make_MS(self):

        self.low = create_named_configuration("LOW-AA0.5")
        self.freqwin = 200
        self.ntimes = 240
        self.times = numpy.linspace(-2.0, +2.0, self.ntimes) * numpy.pi / 12.0
        self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)

        if self.freqwin > 1:
            self.channelwidth = numpy.array(
                self.freqwin * [self.frequency[1] - self.frequency[0]]
            )
        else:
            self.channelwidth = numpy.array([1e6])

        dopol = False
        if dopol:
            self.vis_pol = PolarisationFrame("linear")
            self.image_pol = PolarisationFrame("stokesIQUV")
            f = [1.0, 0.2, 0.0, 0.0]
        else:
            self.vis_pol = PolarisationFrame("stokesI")
            self.image_pol = PolarisationFrame("stokesI")
            f = [1.0]

        flux = numpy.array(self.freqwin * [f])

        self.phasecentre = SkyCoord(
            ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
        )
        pointsource = Skycomponent(
            direction=self.phasecentre,
            polarisation_frame=self.image_pol,
            flux=flux,
            frequency=self.frequency,
        )

        self.bvis_original = ingest_unittest_visibility(
            self.low,
            self.frequency,
            self.channelwidth,
            self.times,
            self.vis_pol,
            self.phasecentre,
        )

        self.bvis_original = dft_skycomponent_visibility(
            self.bvis_original, pointsource
        )

        self.gt = create_gaintable_from_blockvisibility(self.bvis_original)
        self.gt = simulate_gaintable(self.gt, phase_error=0.1)
        self.bvis_error = apply_gaintable(self.bvis_original, self.gt)

        shutil.rmtree(
            rascil_path("test_results/test_rascil_rcal_gaintable.hdf5"),
            ignore_errors=True,
        )
        shutil.rmtree(
            rascil_path("test_results/test_rascil_rcal.ms"), ignore_errors=True
        )
        export_blockvisibility_to_ms(
            rascil_path("test_results/test_rascil_rcal.ms"), [self.bvis_error]
        )

    def setUp(self) -> None:

        parser = cli_parser()
        self.args = parser.parse_args([])
        self.args.ingest_msname = rascil_path("test_results/test_rascil_rcal.ms")

    def tearDown(self) -> None:
        shutil.rmtree(
            rascil_path("test_results/test_rascil_rcal.ms"), ignore_errors=True
        )
        shutil.rmtree(
            rascil_path("test_results/test_rascil_rcal_gaintable.hdf5"),
            ignore_errors=True,
        )

    def test_rcal(self):

        self.make_MS()
        gtfile = rcal_simulator(self.args)
        assert os.path.exists(gtfile)
        newgt = import_gaintable_from_hdf5(gtfile)
        log.info(f"Gaintable: {newgt}")

        bvis_difference = apply_gaintable(self.bvis_error, newgt, inverse=True)
        bvis_difference["vis"] -= self.bvis_original["vis"]
        qa = qa_visibility(bvis_difference)
        assert qa.data["maxabs"] < 1e-12, str(qa)
        assert qa.data["minabs"] < 1e-12, str(qa)


if __name__ == "__main__":
    unittest.main()
