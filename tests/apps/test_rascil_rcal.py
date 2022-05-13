""" Unit test for rascil_rcal app.

"""
import os
import logging
import unittest
import shutil
import glob
import tempfile
import sys

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.apps.rascil_rcal import (
    cli_parser,
    rcal_simulator,
    get_gain_data,
    gt_single_plot,
    read_skycomponent_from_txt_with_external_frequency,
    _rfi_flagger,
    apply_beam_correction,
    realtime_single_bvis_solver,
)
from rascil.data_models import (
    Skycomponent,
    import_gaintable_from_hdf5,
    export_skycomponent_to_hdf5,
    rascil_path,
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
log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))


class TestRASCILRcal(unittest.TestCase):
    def pre_setup(self, dopol=False):
        """Create and fill values into the MeassurementSet

        :param dopol: Use polarisation?
        """

        self.persist = os.getenv("RASCIL_PERSIST", False)

        self.low = create_named_configuration("LOW-AA0.5")
        # For the flagger to work, the number of channels and times should be larger than 32
        # Because the largest sequence number is 32
        self.freqwin = 40
        self.ntimes = 48
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
            self.tempdir + "/test_rascil_rcal_components.hdf",
        )

        self.bvis_error = self.create_apply_gains()

        export_blockvisibility_to_ms(
            self.tempdir + "/test_rascil_rcal.ms", [self.bvis_error]
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

        self.txtfile = self.tempdir + "/test_rascil_rcal_components.txt"

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
        self.gt = create_gaintable_from_blockvisibility(
            self.bvis_original, jones_type="B"
        )
        self.gt = simulate_gaintable(self.gt, phase_error=0.1)
        qa_gt = qa_gaintable(self.gt)
        assert qa_gt.data["rms-amp"] < 1e-12, str(qa_gt)
        assert qa_gt.data["rms-phase"] > 0.0, str(qa_gt)
        bvis_error = apply_gaintable(self.bvis_original, self.gt)
        assert numpy.std(numpy.angle(bvis_error["vis"].data)) > 0.0

        return bvis_error

    def persist_data_files(self):
        """Persist the temporary data files"""
        try:
            shutil.copyfile(
                self.tempdir + "/test_rascil_rcal.ms", rascil_path("test_results")
            )
        except FileNotFoundError:
            pass

        to_copy = self.tempdir + "/test_rascil_rcal*"
        for f in glob.glob(to_copy):
            shutil.copy(f, rascil_path("test_results"))

    def setUp(self) -> None:

        self.tempdir_root = tempfile.TemporaryDirectory(dir=rascil_path("test_results"))
        self.tempdir = self.tempdir_root.name

        parser = cli_parser()
        self.args = parser.parse_args([])
        self.args.ingest_msname = self.tempdir + "/test_rascil_rcal.ms"
        self.args.ingest_components_file = (
            self.tempdir + "/test_rascil_rcal_components.hdf"
        )
        self.args.do_plotting = "False"
        self.args.plot_dir = self.tempdir + "/"
        self.args.use_previous_gaintable = "True"

    # Regression test
    def test_rcal(self):
        self.pre_setup()
        self.makeMS(self.flux)

        gtfile = rcal_simulator(self.bvis_original, self.args)

        # Check that the gaintable exists and is correct by applying it to
        # the corrupted visibility
        assert os.path.exists(gtfile)
        gain_table = import_gaintable_from_hdf5(gtfile)
        assert (
            gain_table["weight"].data != 0
        ).all()  # un-flagged data, all weights are non-zero
        log.info(f"\nFinal gaintable: {gain_table}")

        qa_gt = qa_gaintable(gain_table)
        log.info(qa_gt)
        assert qa_gt.data["rms-phase"] > 0.0, str(qa_gt)

        bvis_difference = apply_gaintable(self.bvis_error, gain_table, inverse=True)
        bvis_difference["vis"] -= self.bvis_original["vis"]
        qa = qa_visibility(bvis_difference)
        assert qa.data["maxabs"] < 1e-12, str(qa)
        assert qa.data["minabs"] < 1e-12, str(qa)

        # Test the plot does not exist
        self.plotfile = self.tempdir + "/test_rascil_rcal_plot.png"
        assert os.path.exists(self.plotfile) is False

        # Test that when we flag RFI, the results are different from
        # when we flag after gains were calculated
        os.remove(gtfile)
        self.args.flag_rfi = "True"  # flag before gains are calculated
        gtfile = rcal_simulator(self.bvis_original, self.args)
        gain_table_w_flag = import_gaintable_from_hdf5(gtfile)

        assert (gain_table_w_flag["weight"].data != gain_table["weight"].data).any()

        if self.persist is True:
            self.persist_data_files()

    def test_rcal_plot(self):
        self.pre_setup()
        self.create_dft_components(self.flux)
        self.bvis_error = self.create_apply_gains()

        self.plotfile = self.tempdir + "/test_rascil_rcal_plot.png"
        plot_name = self.plotfile.replace(".png", "")
        gt_single_plot(self.gt, plot_name=plot_name)

        assert os.path.exists(self.plotfile)

        if self.persist is True:
            self.persist_data_files()

    # Unit tests for additional functions
    def test_read_txtfile(self):
        """Test for read_skycomponent_from_txt_with_external_frequency"""
        self.pre_setup()
        comp = self.create_dft_components(self.flux)
        self.write_to_txt(comp)

        components_read = read_skycomponent_from_txt_with_external_frequency(
            self.txtfile, self.frequency, self.vis_pol
        )
        assert components_read.direction == self.phasecentre
        assert components_read.flux[:, 0].all() == self.flux.all()

        if self.persist is True:
            self.persist_data_files()

    def test_apply_beam_correction(self):
        """Test for apply_beam_correction
        Currently only test for LOW"""

        self.pre_setup()
        new_bvis = self.bvis_original.copy(deep=True)
        comp = self.create_dft_components(self.flux)
        new_comp = apply_beam_correction(new_bvis, [comp], None, telescope_name="LOW")

        assert len(new_comp) == 1
        assert new_comp[0].direction == self.phasecentre
        assert numpy.any(numpy.not_equal(new_comp[0].flux, self.flux))

        if self.persist is True:
            self.persist_data_files()

    def test_get_gain_data(self):
        self.pre_setup()
        self.create_dft_components(self.flux)
        self.bvis_error = self.create_apply_gains()

        gain_data = get_gain_data(self.gt)
        assert len(gain_data[0]) == 1  # time dimension
        assert len(gain_data[1]) == 6  # gain dimension (number of antennas)
        assert len(gain_data[2]) == 6  # phase dimension
        assert len(gain_data[3]) == 1  # residual dimension
        assert len(gain_data[4]) == 6  # weight dimension

        if self.persist is True:
            self.persist_data_files()

    def test_rfi_flagger(self):
        self.pre_setup()
        new_bvis = self.bvis_original.copy(deep=True)
        # update new_bvis to have a value that will be flagged
        new_bvis["vis"].data[0, 0, 0, 0] = complex(100, 0)

        _rfi_flagger(new_bvis)

        assert new_bvis != self.bvis_original
        # the flagger flagged that single data point
        assert new_bvis["flags"].data[0, 0, 0, 0] == 1
        # the flags array is all 0s in the original bvis
        assert (self.bvis_original["flags"] == 0).all()

        if self.persist is True:
            self.persist_data_files()

    def test_realtime_single_bvis_solver(self):
        """
        use_previous=True surfaced a bug in the code;
        this test makes sure that at least the code doesn't
        break when that argument is set.
        """
        self.pre_setup()
        bvis = self.bvis_original.copy(deep=True)
        model_components = None
        previous_solution = None
        use_previous = True

        result_gain_table, previous_solution = realtime_single_bvis_solver(
            bvis, model_components, previous_solution, use_previous=use_previous
        )

        assert previous_solution is not None

        # shape: (time, antenna, frequency, receptor1, receptor2
        # time and frequency are same as in bvis
        assert result_gain_table["gain"].shape == (
            len(bvis.time),
            6,
            len(bvis.frequency),
            1,
            1,
        )
        assert (result_gain_table["gain"].data == 1.0 + 0j).all()

        if self.persist is True:
            self.persist_data_files()


if __name__ == "__main__":
    unittest.main()
