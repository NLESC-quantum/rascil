""" Unit test for rascil_rcal app.

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
from rascil.data_models import (
    rascil_path,
    Skycomponent,
    import_gaintable_from_hdf5,
    export_skycomponent_to_hdf5,
)
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import (
    export_blockvisibility_to_ms,
    dft_skycomponent_visibility,
    create_gaintable_from_blockvisibility,
    simulate_gaintable,
    apply_gaintable,
    qa_visibility,
    qa_gaintable,
)
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.simulation import ingest_unittest_visibility

log = logging.getLogger("rascil-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


class TestRASCILRcal(unittest.TestCase):
    def make_MS(self, dopol=False):
        """Create and fill values into the MeassurementSet

        :param dopol: Use polarisation?
        """

        self.cleanup_data_files()

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

        if dopol:
            self.vis_pol = PolarisationFrame("linear")
            self.image_pol = PolarisationFrame("stokesIQUV")
            f = [100.0, 20.0, 0.0, 0.0]
        else:
            self.vis_pol = PolarisationFrame("stokesI")
            self.image_pol = PolarisationFrame("stokesI")
            f = [100.0]

        flux = numpy.array(self.freqwin * [f])

        self.phasecentre = SkyCoord(
            ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.bvis_original = ingest_unittest_visibility(
            self.low,
            self.frequency,
            self.channelwidth,
            self.times,
            self.vis_pol,
            self.phasecentre,
        )

        self.create_dft_components(flux)

        self.create_apply_gains()

    def create_dft_components(self, flux):
        """Create the components, save to file, dft into visibility

        :param flux:
        """
        pointsource = Skycomponent(
            direction=self.phasecentre,
            polarisation_frame=self.image_pol,
            flux=flux,
            frequency=self.frequency,
        )
        self.bvis_original = dft_skycomponent_visibility(
            self.bvis_original, pointsource
        )
        export_skycomponent_to_hdf5(
            [pointsource],
            rascil_path("test_results/test_rascil_rcal_components.hdf"),
        )

    def create_apply_gains(self):
        """Create the gaintable, apply to the visibility, write as MeasurementSet

        :return:
        """
        self.gt = create_gaintable_from_blockvisibility(self.bvis_original)
        self.gt = simulate_gaintable(self.gt, phase_error=0.1)
        qa_gt = qa_gaintable(self.gt)
        assert qa_gt.data["rms-amp"] < 1e-12, str(qa_gt)
        assert qa_gt.data["rms-phase"] > 0.0, str(qa_gt)
        self.bvis_error = apply_gaintable(self.bvis_original, self.gt)
        assert numpy.std(numpy.angle(self.bvis_error["vis"].data)) > 0.0
        export_blockvisibility_to_ms(
            rascil_path("test_results/test_rascil_rcal.ms"), [self.bvis_error]
        )

    def cleanup_data_files(self):
        """Cleanup the temporary data files"""
        shutil.rmtree(
            rascil_path("test_results/test_rascil_rcal_components.hdf"),
            ignore_errors=True,
        )
        shutil.rmtree(
            rascil_path("test_results/test_rascil_rcal_gaintable.hdf"),
            ignore_errors=True,
        )
        shutil.rmtree(
            rascil_path("test_results/test_rascil_rcal.ms"), ignore_errors=True
        )

    def setUp(self) -> None:

        parser = cli_parser()
        self.args = parser.parse_args([])
        self.args.ingest_msname = rascil_path("test_results/test_rascil_rcal.ms")
        self.args.ingest_components_file = rascil_path(
            "test_results/test_rascil_rcal_components.hdf"
        )
        self.args.do_plotting = "True"
        self.args.plot_dir = rascil_path("test_results/")

    def tearDown(self) -> None:
        self.cleanup_data_files()

    def test_rcal(self):

        self.make_MS()
        gtfile = rcal_simulator(self.args)

        # Check that the gaintable exists and is correct by applying it to
        # the corrupted visibility
        assert os.path.exists(gtfile)
        newgt = import_gaintable_from_hdf5(gtfile)
        log.info(f"\nFinal gaintable: {newgt}")

        qa_gt = qa_gaintable(newgt)
        log.info(qa_gt)
        assert qa_gt.data["rms-phase"] > 0.0, str(qa_gt)

        bvis_difference = apply_gaintable(self.bvis_error, newgt, inverse=True)
        bvis_difference["vis"] -= self.bvis_original["vis"]
        qa = qa_visibility(bvis_difference)
        assert qa.data["maxabs"] < 1e-12, str(qa)
        assert qa.data["minabs"] < 1e-12, str(qa)

        plotfile = rascil_path("test_results/test_rascil_rcal_plot.png")
        assert os.path.exists(plotfile)


if __name__ == "__main__":
    unittest.main()
