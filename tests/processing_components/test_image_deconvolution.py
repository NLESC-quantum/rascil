""" Unit tests for image deconvolution


"""
import os
import logging
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.data_models import Skycomponent

from rascil.processing_components.arrays.cleaners import overlapIndices
from rascil.processing_components.image.deconvolution import (
    hogbom_kernel_list,
    find_window_list,
)
from rascil.processing_components.skycomponent.operations import restore_skycomponent

from rascil.processing_components import (
    restore_list,
    deconvolve_cube,
    restore_cube,
    fit_psf,
    create_pb,
)
from rascil.processing_components.image.operations import export_image_to_fits, qa_image
from rascil.processing_components.simulation import create_test_image
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility.base import create_blockvisibility
from rascil.processing_components.imaging.imaging import (
    predict_blockvisibility,
    invert_blockvisibility,
)
from rascil.processing_components.imaging.base import (
    create_image_from_visibility,
)

log = logging.getLogger("rascil-logger")

log.setLevel(logging.INFO)


class TestImageDeconvolution(unittest.TestCase):
    def setUp(self):

        self.persist = os.getenv("RASCIL_PERSIST", False)

        from rascil.data_models.parameters import rascil_path, rascil_data_path

        self.results_dir = rascil_path("test_results")
        self.lowcore = create_named_configuration("LOWBD2-CORE")
        self.times = (numpy.pi / (12.0)) * numpy.linspace(-3.0, 3.0, 7)
        self.frequency = numpy.array([1e8])
        self.channel_bandwidth = numpy.array([1e6])
        self.phasecentre = SkyCoord(
            ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.vis = create_blockvisibility(
            self.lowcore,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            phasecentre=self.phasecentre,
            weight=1.0,
            polarisation_frame=PolarisationFrame("stokesI"),
            zerow=True,
        )
        self.vis["vis"].data *= 0.0

        # Create model
        self.test_model = create_test_image(
            cellsize=0.001, frequency=self.frequency, phasecentre=self.vis.phasecentre
        )
        self.vis = predict_blockvisibility(self.vis, self.test_model, context="2d")
        assert numpy.max(numpy.abs(self.vis.vis)) > 0.0
        self.model = create_image_from_visibility(
            self.vis,
            npixel=512,
            cellsize=0.001,
            polarisation_frame=PolarisationFrame("stokesI"),
        )
        self.dirty = invert_blockvisibility(self.vis, self.model, context="2d")[0]
        self.psf = invert_blockvisibility(
            self.vis, self.model, context="2d", dopsf=True
        )[0]
        self.sensitivity = create_pb(self.model, "LOW")

    def overlaptest(self, a1, a2, s1, s2):
        #
        a1[s1[0] : s1[1], s1[2] : s1[3]] = 1
        a2[s2[0] : s2[1], s2[2] : s2[3]] = 1
        return numpy.sum(a1) == numpy.sum(a2)

    def test_overlap(self):
        res = numpy.zeros([512, 512])
        psf = numpy.zeros([100, 100])
        peak = (499, 249)
        s1, s2 = overlapIndices(res, psf, peak[0], peak[1])
        assert len(s1) == 4
        assert len(s2) == 4
        self.overlaptest(res, psf, s1, s2)
        assert s1 == (449, 512, 199, 299)
        assert s2 == (0, 63, 0, 100)

    def test_restore(self):
        self.model["pixels"].data[0, 0, 256, 256] = 1.0
        self.cmodel = restore_cube(self.model, self.psf)
        assert numpy.abs(numpy.max(self.cmodel["pixels"].data) - 1.0) < 1e-7, numpy.max(
            self.cmodel["pixels"].data
        )
        if self.persist:
            export_image_to_fits(
                self.cmodel, "%s/test_restore.fits" % (self.results_dir)
            )

    def test_restore_list(self):
        self.model["pixels"].data[0, 0, 256, 256] = 1.0
        self.cmodel = restore_list([self.model], [self.psf])[0]
        assert numpy.abs(numpy.max(self.cmodel["pixels"].data) - 1.0) < 1e-7, numpy.max(
            self.cmodel["pixels"].data
        )
        if self.persist:
            export_image_to_fits(
                self.cmodel, "%s/test_restore.fits" % (self.results_dir)
            )

    def test_restore_clean_beam(self):
        """Test restoration with specified beam beam

        :return:
        """
        self.model["pixels"].data[0, 0, 256, 256] = 1.0
        # The beam is specified in degrees
        bmaj = 0.006 * 180.0 / numpy.pi
        self.cmodel = restore_cube(
            self.model,
            self.psf,
            clean_beam={"bmaj": bmaj, "bmin": bmaj, "bpa": 0.0},
        )
        assert numpy.abs(numpy.max(self.cmodel["pixels"].data) - 1.0) < 1e-7, numpy.max(
            self.cmodel["pixels"].data
        )
        if self.persist:
            export_image_to_fits(
                self.cmodel, "%s/test_restore_6mrad_beam.fits" % (self.results_dir)
            )

    def test_restore_skycomponent(self):
        """Test restoration of single pixel and skycomponent"""
        self.model["pixels"].data[0, 0, 256, 256] = 0.5

        sc = Skycomponent(
            flux=numpy.array([[1.0]]),
            direction=SkyCoord(
                ra=+180.0 * u.deg, dec=-61.0 * u.deg, frame="icrs", equinox="J2000"
            ),
            shape="Point",
            frequency=self.frequency,
            polarisation_frame=PolarisationFrame("stokesI"),
        )
        bmaj = 0.012 * 180.0 / numpy.pi
        clean_beam = {"bmaj": bmaj, "bmin": bmaj / 2.0, "bpa": 15.0}
        self.cmodel = restore_cube(self.model, clean_beam=clean_beam)
        self.cmodel = restore_skycomponent(self.cmodel, sc, clean_beam=clean_beam)
        if self.persist:
            export_image_to_fits(
                self.cmodel, "%s/test_restore_skycomponent.fits" % (self.results_dir)
            )
        assert (
            numpy.abs(numpy.max(self.cmodel["pixels"].data) - 0.9959046879055156) < 1e-7
        ), numpy.max(self.cmodel["pixels"].data)

    def test_fit_psf(self):
        clean_beam = fit_psf(self.psf)
        if self.persist:
            export_image_to_fits(self.psf, "%s/test_fit_psf.fits" % (self.results_dir))
        # Sanity check: by eyeball the FHWM = 4 pixels = 0.004 rad = 0.229 deg
        assert numpy.abs(clean_beam["bmaj"] - 0.24790689057765794) < 1.0e-7, clean_beam
        assert numpy.abs(clean_beam["bmin"] - 0.2371401153972545) < 1.0e-7, clean_beam
        assert numpy.abs(clean_beam["bpa"] + 1.0126425267576473) < 1.0e-7, clean_beam

    def test_deconvolve_hogbom(self):
        self.comp, self.residual = deconvolve_cube(
            self.dirty,
            self.psf,
            niter=10000,
            gain=0.1,
            algorithm="hogbom",
            threshold=0.01,
        )
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        if self.persist:
            self.save_results("hogbom")
        assert numpy.max(self.residual["pixels"].data) < 1.2

    def test_deconvolve_msclean(self):
        self.comp, self.residual = deconvolve_cube(
            self.dirty,
            self.psf,
            niter=1000,
            gain=0.7,
            algorithm="msclean",
            scales=[0, 3, 10, 30],
            threshold=0.01,
        )
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        if self.persist:
            self.save_results("msclean")
        assert numpy.max(self.residual["pixels"].data) < 1.2

    def save_results(self, tag):
        export_image_to_fits(
            self.comp, f"{self.results_dir}/test_deconvolve_{tag}-deconvolved.fits"
        )
        export_image_to_fits(
            self.residual,
            f"{self.results_dir}/test_deconvolve_{tag}-residual.fits",
        )
        export_image_to_fits(
            self.cmodel, f"{self.results_dir}/test_deconvolve_{tag}-restored.fits"
        )

    def test_deconvolve_msclean_sensitivity(self):
        self.comp, self.residual = deconvolve_cube(
            self.dirty,
            self.psf,
            sensitivity=self.sensitivity,
            niter=1000,
            gain=0.7,
            algorithm="msclean",
            scales=[0, 3, 10, 30],
            threshold=0.01,
        )
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        if self.persist:
            self.save_results("msclean-sensitivity")

        qa = qa_image(self.residual)
        numpy.testing.assert_allclose(
            qa.data["max"], 0.8040729590477751, atol=1e-7, err_msg=f"{qa}"
        )
        numpy.testing.assert_allclose(
            qa.data["min"], -0.9044553283128349, atol=1e-7, err_msg=f"{qa}"
        )

    def test_deconvolve_msclean_1scale(self):

        self.comp, self.residual = deconvolve_cube(
            self.dirty,
            self.psf,
            niter=10000,
            gain=0.1,
            algorithm="msclean",
            scales=[0],
            threshold=0.01,
        )
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        if self.persist:
            self.save_results("msclean-1scale")
        assert numpy.max(self.residual["pixels"].data) < 1.2

    def test_deconvolve_hogbom_no_edge(self):
        self.comp, self.residual = deconvolve_cube(
            self.dirty,
            self.psf,
            window_shape="no_edge",
            niter=10000,
            gain=0.1,
            algorithm="hogbom",
            threshold=0.01,
        )
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        if self.persist:
            self.save_results("hogbom_no_edge")
        assert numpy.max(self.residual["pixels"].data) < 1.2

    def test_deconvolve_hogbom_inner_quarter(self):
        self.comp, self.residual = deconvolve_cube(
            self.dirty,
            self.psf,
            window_shape="quarter",
            niter=10000,
            gain=0.1,
            algorithm="hogbom",
            threshold=0.01,
        )
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        if self.persist:
            self.save_results("hogbom_no_inner_quarter")
        assert numpy.max(self.residual["pixels"].data) < 1.2

    def test_deconvolve_msclean_inner_quarter(self):

        self.comp, self.residual = deconvolve_cube(
            self.dirty,
            self.psf,
            window_shape="quarter",
            niter=1000,
            gain=0.7,
            algorithm="msclean",
            scales=[0, 3, 10, 30],
            threshold=0.01,
        )
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        if self.persist:
            self.save_results("msclean_inner_quarter")
        assert numpy.max(self.residual["pixels"].data) < 1.2

    def test_deconvolve_hogbom_subpsf(self):

        self.comp, self.residual = deconvolve_cube(
            self.dirty,
            psf=self.psf,
            psf_support=200,
            window_shape="quarter",
            niter=10000,
            gain=0.1,
            algorithm="hogbom",
            threshold=0.01,
        )
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        if self.persist:
            self.save_results("hogbom_subpsf")
        assert numpy.max(self.residual["pixels"].data[..., 56:456, 56:456]) < 1.2

    def test_deconvolve_msclean_subpsf(self):

        self.comp, self.residual = deconvolve_cube(
            self.dirty,
            psf=self.psf,
            psf_support=200,
            window_shape="quarter",
            niter=1000,
            gain=0.7,
            algorithm="msclean",
            scales=[0, 3, 10, 30],
            threshold=0.01,
        )
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        if self.persist:
            self.save_results("msclean_subpsf")
        assert numpy.max(self.residual["pixels"].data[..., 56:456, 56:456]) < 1.0

    def _check_hogbom_kernel_list_test_results(self, component, residual):
        result_comp_data = component["pixels"].data
        non_zero_idx_comp = numpy.where(result_comp_data != 0.0)
        expected_comp_non_zero_data = numpy.array(
            [
                0.508339,
                0.590298,
                0.533506,
                0.579212,
                0.549127,
                0.622576,
                0.538019,
                0.717473,
                0.716564,
                0.840854,
            ]
        )
        result_residual_data = residual["pixels"].data
        non_zero_idx_residual = numpy.where(result_residual_data != 0.0)
        expected_residual_non_zero_data = numpy.array(
            [
                0.214978,
                0.181119,
                0.145942,
                0.115912,
                0.100664,
                0.106727,
                0.132365,
                0.167671,
                0.200349,
                0.222765,
            ]
        )

        # number of non-zero values
        assert len(result_comp_data[non_zero_idx_comp]) == 82
        assert len(result_residual_data[non_zero_idx_residual]) == 262144
        # test first 10 non-zero values don't change with each run of test
        numpy.testing.assert_array_almost_equal(
            result_comp_data[non_zero_idx_comp][:10], expected_comp_non_zero_data
        )
        numpy.testing.assert_array_almost_equal(
            result_residual_data[non_zero_idx_residual][:10],
            expected_residual_non_zero_data,
        )

    def test_hogbom_kernel_list_single_dirty(self):
        prefix = "test_hogbom_list"
        dirty_list = [self.dirty]
        psf_list = [self.psf]
        window_list = find_window_list(dirty_list, prefix)

        comp_list, residual_list = hogbom_kernel_list(
            dirty_list, prefix, psf_list, window_list
        )

        assert len(comp_list) == 1
        assert len(residual_list) == 1
        self._check_hogbom_kernel_list_test_results(comp_list[0], residual_list[0])

    def test_hogbom_kernel_list_multiple_dirty(self, window_shape=None):
        """
        Bugfix: hogbom_kernel_list produced an IndexError, when dirty_list has more than
        one elements, and those elements are for a single frequency each, and window_shape is None.
        """

        prefix = "test_hogbom_list"
        dirty_list = [self.dirty, self.dirty]
        psf_list = [self.psf, self.psf]
        window_list = find_window_list(dirty_list, prefix, window_shape)

        comp_list, residual_list = hogbom_kernel_list(
            dirty_list, prefix, psf_list, window_list
        )

        assert len(comp_list) == 2
        assert len(residual_list) == 2
        # because the two dirty images and psfs are the same, the expected results are also the same
        self._check_hogbom_kernel_list_test_results(comp_list[0], residual_list[0])
        self._check_hogbom_kernel_list_test_results(comp_list[1], residual_list[1])

    def test_hogbom_kernel_list_multiple_dirty_window_shape(self):
        """
        Buffix: hogbom_kernel_list produced an IndexError. Test the second branch of the if statement
        when dirty_list has more than one elements.
        """
        self.test_hogbom_kernel_list_multiple_dirty(window_shape="quarter")


if __name__ == "__main__":
    unittest.main()
