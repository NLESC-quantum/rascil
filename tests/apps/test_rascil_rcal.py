""" Unit test for rascil_rcal app.

"""
import sys
import os
import logging
import unittest
import shutil
import glob

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.visualization import time_support

from rascil.apps.rascil_rcal import (
    cli_parser,
    rcal_simulator,
    get_gain_data,
    read_skycomponent_from_txt_with_external_frequency,
)
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
    def pre_setup(self, dopol=False):
        """Create and fill values into the MeassurementSet

        :param dopol: Use polarisation?
        """

        self.persist = os.getenv("RASCIL_PERSIST", False)

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

        self.flux = numpy.array(self.freqwin * [f])

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

    def makeMS(self, flux):

        comp = self.create_dft_components(flux)

        export_skycomponent_to_hdf5(
            [comp],
            rascil_path("test_results/test_rascil_rcal_components.hdf"),
        )

        self.bvis_error = self.create_apply_gains()

        export_blockvisibility_to_ms(
            rascil_path("test_results/test_rascil_rcal.ms"), [self.bvis_error]
        )

    def create_dft_components(self, flux):
        """Create the components, save to file, dft into visibility

        :param flux:

        :return pointsource: Point source skycomponent
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

        return pointsource

    def write_to_txt(self, comp):

        self.txtfile = rascil_path("test_results/test_rascil_rcal_components.txt")

        coord_ra = comp.direction.ra.degree
        coord_dec = comp.direction.dec.degree
        f = open(self.txtfile, "w")
        f.write(
            "%.6f, %.6f, %10.6e, %10.6e, %10.6e, %10.6e\n"
            % (
                coord_ra,
                coord_dec,
                comp.flux[0][0],
                0.0,
                0.0,
                0.0,
            )
        )
        f.close()

    def create_apply_gains(self):
        """Create the gaintable, apply to the visibility, write as MeasurementSet

        :return: bvis_error: BlockVisibility
        """
        self.gt = create_gaintable_from_blockvisibility(self.bvis_original)
        self.gt = simulate_gaintable(self.gt, phase_error=0.1)
        qa_gt = qa_gaintable(self.gt)
        assert qa_gt.data["rms-amp"] < 1e-12, str(qa_gt)
        assert qa_gt.data["rms-phase"] > 0.0, str(qa_gt)
        bvis_error = apply_gaintable(self.bvis_original, self.gt)
        assert numpy.std(numpy.angle(bvis_error["vis"].data)) > 0.0

        return bvis_error

    def cleanup_data_files(self):
        """Cleanup the temporary data files"""

        # First remove the measurement set
        shutil.rmtree(
            rascil_path("test_results/test_rascil_rcal.ms"), ignore_errors=True
        )

        to_remove = rascil_path("test_results/test_rascil_rcal*")
        for f in glob.glob(to_remove):
            if os.path.exists(f):
                os.remove(f)

    def setUp(self) -> None:

        parser = cli_parser()
        self.args = parser.parse_args([])
        self.args.ingest_msname = rascil_path("test_results/test_rascil_rcal.ms")
        self.args.ingest_components_file = rascil_path(
            "test_results/test_rascil_rcal_components.hdf"
        )
        self.args.do_plotting = "False"
        self.args.plot_dir = rascil_path("test_results/")

    # Regression test
    def test_rcal(self):

        self.pre_setup()
        self.makeMS(self.flux)
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

        # Test the plotting
        plotfile = rascil_path("test_results/test_rascil_rcal_plot.png")
        assert os.path.exists(plotfile) == False

        self.args.do_plotting = "True"
        gtfile_new = rcal_simulator(self.args)
        assert os.path.exists(plotfile)

        if self.persist is False:
            self.cleanup_data_files()

        # Unit tests for additional functions

    def test_read_txtfile(self):
        "Test for read_skycomponent_from_txt_with_external_frequency"

        self.pre_setup()
        comp = self.create_dft_components(self.flux)
        self.write_to_txt(comp)

        components_read = read_skycomponent_from_txt_with_external_frequency(
            self.txtfile, self.frequency, self.vis_pol
        )
        assert components_read.direction == self.phasecentre
        assert components_read.flux[:, 0].all() == self.flux.all()

        if self.persist is False:
            self.cleanup_data_files()

    def test_get_gain_data(self):

        self.pre_setup()
        comp = self.create_dft_components(self.flux)
        self.bvis_error = self.create_apply_gains()

        gain_data = get_gain_data(self.gt)
        assert len(gain_data[0]) == 1  # time dimension
        assert len(gain_data[1]) == 6  # gain dimension (number of antennas)
        assert len(gain_data[2]) == 6  # phase dimension
        assert len(gain_data[3]) == 1  # residual dimension

        if self.persist is False:
            self.cleanup_data_files()


if __name__ == "__main__":
    unittest.main()
