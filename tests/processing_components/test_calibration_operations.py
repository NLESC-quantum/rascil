""" Unit tests for visibility calibration


"""

import numpy
import logging
import unittest

from astropy.coordinates import SkyCoord
import astropy.units as u

from rascil.data_models.memory_data_models import Skycomponent
from rascil.data_models.polarisation import PolarisationFrame

from rascil.processing_components import (
    gaintable_summary,
    apply_gaintable,
    create_gaintable_from_blockvisibility,
    concatenate_gaintables,
)
from rascil.processing_components.simulation import simulate_gaintable
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility.base import (
    copy_visibility,
    create_blockvisibility,
)
from rascil.processing_components.imaging import dft_skycomponent_visibility

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)


class TestCalibrationOperations(unittest.TestCase):
    def setUp(self):
        pass

    def actualSetup(self, sky_pol_frame="stokesIQUV", data_pol_frame="linear"):
        self.lowcore = create_named_configuration("LOWBD2", rmax=100.0)
        self.times = (numpy.pi / 43200.0) * numpy.arange(0.0, 3000.0, 60.0)
        vnchan = 3
        self.frequency = numpy.linspace(1.0e8, 1.1e8, vnchan)
        self.channel_bandwidth = numpy.array(
            vnchan * [self.frequency[1] - self.frequency[0]]
        )

        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])

        self.phasecentre = SkyCoord(
            ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.compabsdirection = SkyCoord(
            ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
        )
        if sky_pol_frame == "stokesI":
            self.flux = self.flux[:, 0][:, numpy.newaxis]

        self.comp = Skycomponent(
            direction=self.compabsdirection,
            frequency=self.frequency,
            flux=self.flux,
            polarisation_frame=PolarisationFrame(sky_pol_frame),
        )
        self.vis = create_blockvisibility(
            self.lowcore,
            self.times,
            self.frequency,
            phasecentre=self.phasecentre,
            channel_bandwidth=self.channel_bandwidth,
            weight=1.0,
            polarisation_frame=PolarisationFrame(data_pol_frame),
        )
        self.vis = dft_skycomponent_visibility(self.vis, self.comp)

    def test_create_gaintable_from_visibility(self):
        """Create the gaintable"""
        for jones_type in ["T", "G", "B"]:
            for spf, dpf in [
                ("stokesI", "stokesI"),
                ("stokesIQUV", "linear"),
                ("stokesIQUV", "circular"),
            ]:
                self.actualSetup(spf, dpf)
                gt = create_gaintable_from_blockvisibility(
                    self.vis, timeslice="auto", jones_type=jones_type
                )
                log.info("Created gain table: %s" % (gaintable_summary(gt)))
                gt = simulate_gaintable(gt, phase_error=1.0)
                original = copy_visibility(self.vis)
                vis = apply_gaintable(self.vis, gt)
                corrected_vis = apply_gaintable(vis, gt, inverse=True)
                assert (
                    numpy.max(numpy.abs(corrected_vis.vis.data - original.vis.data))
                    < 1e-12
                )

    def test_create_gaintable_from_visibility_interval(self):
        """Apply the gaintable and inverse for different values of timeslice"""
        for jones_type in ["T", "G", "B"]:
            for timeslice in [10.0, "auto", 1e5]:
                for spf, dpf in [
                    ("stokesI", "stokesI"),
                    ("stokesIQUV", "linear"),
                    ("stokesIQUV", "circular"),
                ]:
                    self.actualSetup(spf, dpf)
                    gt = create_gaintable_from_blockvisibility(
                        self.vis, timeslice=timeslice, jones_type=jones_type
                    )
                    log.info("Created gain table: %s" % (gaintable_summary(gt)))
                    gt = simulate_gaintable(gt, phase_error=1.0)
                    original = copy_visibility(self.vis)
                    vis = apply_gaintable(self.vis, gt)
                    corrected_vis = apply_gaintable(vis, gt, inverse=True)
                    assert (
                        numpy.max(numpy.abs(corrected_vis.vis.data - original.vis.data))
                        < 1e-12
                    )

    def test_apply_gaintable_only(self):
        """Does applying the gaintable change the visibility?"""
        for jones_type in ["T", "G", "B"]:
            for spf, dpf in [
                ("stokesI", "stokesI"),
                ("stokesIQUV", "linear"),
                ("stokesIQUV", "circular"),
            ]:
                self.actualSetup(spf, dpf)
                gt = create_gaintable_from_blockvisibility(
                    self.vis, timeslice="auto", jones_type=jones_type
                )
                log.info("Created gain table: %s" % (gaintable_summary(gt)))
                gt = simulate_gaintable(gt, phase_error=0.1, amplitude_error=0.01)
                original = copy_visibility(self.vis)
                vis = apply_gaintable(self.vis, gt)
                assert numpy.max(numpy.abs(vis.vis.data - original.vis.data)) > 0.0

    def test_apply_gaintable_and_inverse_phase_only(self):
        for jones_type in ["T", "G", "B"]:
            for spf, dpf in [
                ("stokesI", "stokesI"),
                ("stokesIQUV", "linear"),
                ("stokesIQUV", "circular"),
            ]:
                self.actualSetup(spf, dpf)
                gt = create_gaintable_from_blockvisibility(
                    self.vis, timeslice="auto", jones_type=jones_type
                )
                log.info("Created gain table: %s" % (gaintable_summary(gt)))
                gt = simulate_gaintable(gt, phase_error=0.1)
                original = copy_visibility(self.vis)
                vis = apply_gaintable(self.vis, gt)
                corrected_vis = apply_gaintable(vis, gt, inverse=True)
                assert (
                    numpy.max(numpy.abs(corrected_vis.vis.data - original.vis.data))
                    < 1e-12
                )

    def test_apply_gaintable_and_inverse_both(self):
        for jones_type in ["T", "G", "B"]:
            for spf, dpf in [
                ("stokesI", "stokesI"),
                ("stokesIQUV", "linear"),
                ("stokesIQUV", "circular"),
            ]:
                self.actualSetup(spf, dpf)
                gt = create_gaintable_from_blockvisibility(
                    self.vis, timeslice="auto", jones_type=jones_type
                )
                log.info("Created gain table: %s" % (gaintable_summary(gt)))
                gt = simulate_gaintable(gt, phase_error=0.1, amplitude_error=0.1)
                original = copy_visibility(self.vis)
                vis = apply_gaintable(self.vis, gt)
                corrected_vis = apply_gaintable(vis, gt, inverse=True)
                assert (
                    numpy.max(numpy.abs(corrected_vis.vis.data - original.vis.data))
                    < 1e-12
                )

    def test_apply_gaintable_null(self):
        """Check if applying zero gains is a no-op"""
        for jones_type in ["T", "G", "B"]:
            for spf, dpf in [
                ("stokesI", "stokesI"),
                ("stokesIQUV", "linear"),
                ("stokesIQUV", "circular"),
            ]:
                self.actualSetup(spf, dpf)
                gt = create_gaintable_from_blockvisibility(
                    self.vis, timeslice="auto", jones_type=jones_type
                )
                gt["gain"].data *= 0.0
                vis = apply_gaintable(self.vis, gt)
                error = numpy.max(numpy.abs(vis["vis"].data - self.vis["vis"].data))
                assert error < 1e-12, f"{spf} {dpf} Error = {error}"

    def test_concatenate_gaintable(self):
        pol_frame = "stokesI"
        self.actualSetup(pol_frame, pol_frame)

        new_times = (numpy.pi / 43200.0) * numpy.arange(3001.0, 6000.0, 60.0)
        new_vis = create_blockvisibility(
            self.lowcore,
            new_times,
            self.frequency,
            phasecentre=self.phasecentre,
            channel_bandwidth=self.channel_bandwidth,
            weight=1.0,
            polarisation_frame=PolarisationFrame(pol_frame),
        )

        gt = create_gaintable_from_blockvisibility(
            self.vis, timeslice="auto", jones_type="T"
        )
        new_gt = create_gaintable_from_blockvisibility(
            new_vis, timeslice="auto", jones_type="T"
        )

        combined_gt = concatenate_gaintables([gt, new_gt])

        assert combined_gt.time.size == self.vis.time.size + new_vis.time.size
        assert numpy.isin(self.vis.time, combined_gt.time).all()
        assert numpy.isin(new_vis.time, combined_gt.time).all()


if __name__ == "__main__":
    unittest.main()
