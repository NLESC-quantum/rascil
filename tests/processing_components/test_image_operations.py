""" Unit tests for image operations

"""
import logging
import os
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models.parameters import rascil_data_path
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import (
    create_image,
    create_image_from_array,
    polarisation_frame_from_wcs,
    create_empty_image_like,
    fft_image_to_griddata,
    pad_image,
    sub_image,
    create_w_term_like,
    create_vp,
    apply_voltage_pattern_to_image,
)
from rascil.processing_components.image.operations import (
    export_image_to_fits,
    calculate_image_frequency_moments,
    calculate_image_taylor_terms,
    calculate_image_from_frequency_moments,
    add_image,
    qa_image,
    reproject_image,
    convert_polimage_to_stokes,
    import_image_from_fits,
    average_image_over_frequency,
    convert_stokes_to_polimage,
    smooth_image,
    scale_and_rotate_image,
    fft_image_to_griddata,
    ifft_griddata_to_image,
)
from rascil.processing_components.simulation import (
    create_test_image,
    create_low_test_image_from_gleam,
)

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)


class TestImage(unittest.TestCase):
    def setUp(self):

        from rascil.data_models.parameters import rascil_path

        self.dir = rascil_path("test_results")

        self.m31image = create_test_image()

        # assert numpy.max(self.m31image["pixels"]) > 0.0, "Test image is empty"
        self.cellsize = 180.0 * 0.0001 / numpy.pi
        self.persist = os.getenv("RASCIL_PERSIST", False)

    def test_create_image(self):
        phasecentre = SkyCoord(
            ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
        )
        newimage = create_image(
            npixel=1024,
            cellsize=0.0001,
            polarisation_frame=PolarisationFrame("stokesIQUV"),
            frequency=numpy.linspace(0.8e9, 1.2e9, 5),
            phasecentre=phasecentre,
            channel_bandwidth=1e7 * numpy.ones([5]),
        )
        assert newimage.image_acc.shape == (5, 4, 1024, 1024)
        assert newimage.image_acc.phasecentre.separation(phasecentre).value < 1e-12
        assert newimage.image_acc.polarisation_frame.type == "stokesIQUV"

    def test_create_image_IQ(self):
        phasecentre = SkyCoord(
            ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
        )
        newimage = create_image(
            npixel=1024,
            cellsize=0.0001,
            polarisation_frame=PolarisationFrame("stokesIQ"),
            frequency=numpy.linspace(0.8e9, 1.2e9, 5),
            phasecentre=phasecentre,
            channel_bandwidth=1e7 * numpy.ones([5]),
        )
        assert newimage.image_acc.shape == (5, 2, 1024, 1024)
        assert newimage.image_acc.phasecentre.separation(phasecentre).value < 1e-12
        assert newimage.image_acc.polarisation_frame.type == "stokesIQ"

    def test_create_image_from_array(self):
        clean_beam = {"bmaj": 0.1, "bmin": 0.05, "bpa": -60.0}
        m31model_by_array = create_image_from_array(
            self.m31image["pixels"],
            self.m31image.image_acc.wcs,
            self.m31image.image_acc.polarisation_frame,
            clean_beam,
        )
        export_image_to_fits(
            m31model_by_array, fitsfile="%s/test_model.fits" % (self.dir)
        )
        m31image_by_fits = import_image_from_fits(
            fitsfile="%s/test_model.fits" % (self.dir)
        )
        new_clean_beam = m31image_by_fits.attrs["clean_beam"]
        assert new_clean_beam == clean_beam, new_clean_beam

        log.debug(qa_image(m31model_by_array, context="test_create_from_image"))

    def test_create_image_from_array_raises(self):
        with self.assertRaises(KeyError):
            clean_beam = {"bmaj": 0.1, "bmin": 0.05}
            m31model_by_array = create_image_from_array(
                self.m31image["pixels"],
                self.m31image.image_acc.wcs,
                self.m31image.image_acc.polarisation_frame,
                clean_beam,
            )

    def test_create_empty_image_like(self):
        emptyimage = create_empty_image_like(self.m31image)
        assert emptyimage["pixels"].shape == self.m31image["pixels"].shape
        assert numpy.max(numpy.abs(emptyimage["pixels"].data)) == 0.0

    def test_reproject(self):
        # Reproject an image
        cellsize = 1.5 * self.cellsize
        newwcs = self.m31image.image_acc.wcs.deepcopy()
        newwcs.wcs.cdelt[0] = -cellsize
        newwcs.wcs.cdelt[1] = +cellsize

        newshape = numpy.array(self.m31image["pixels"].data.shape)
        newshape[2] /= 1.5
        newshape[3] /= 1.5
        newimage, footprint = reproject_image(self.m31image, newwcs, shape=newshape)

    def test_stokes_conversion(self):
        assert self.m31image.image_acc.polarisation_frame == PolarisationFrame(
            "stokesI"
        )
        stokes = create_test_image(
            cellsize=0.0001, polarisation_frame=PolarisationFrame("stokesIQUV")
        )
        assert stokes.image_acc.polarisation_frame == PolarisationFrame("stokesIQUV")

        for pol_name in ["circular", "linear"]:
            polarisation_frame = PolarisationFrame(pol_name)
            polimage = convert_stokes_to_polimage(
                stokes, polarisation_frame=polarisation_frame
            )
            assert polimage.image_acc.polarisation_frame == polarisation_frame
            polarisation_frame_from_wcs(
                polimage.image_acc.wcs, polimage.image_acc.shape
            )
            rstokes = convert_polimage_to_stokes(polimage)
            assert polimage["pixels"].data.dtype == "complex"
            assert rstokes["pixels"].data.dtype == "float"
            numpy.testing.assert_array_almost_equal(
                stokes["pixels"].data, rstokes["pixels"].data.real, 12
            )

    def test_polarisation_frame_from_wcs(self):
        assert self.m31image.image_acc.polarisation_frame == PolarisationFrame(
            "stokesI"
        )
        stokes = create_test_image(
            cellsize=0.0001, polarisation_frame=PolarisationFrame("stokesIQUV")
        )
        wcs = stokes.image_acc.wcs.deepcopy()
        shape = stokes.image_acc.shape
        assert polarisation_frame_from_wcs(wcs, shape) == PolarisationFrame(
            "stokesIQUV"
        )

        wcs = stokes.image_acc.wcs.deepcopy().sub(["stokes"])
        wcs.wcs.crpix[0] = 1.0
        wcs.wcs.crval[0] = -1.0
        wcs.wcs.cdelt[0] = -1.0
        assert polarisation_frame_from_wcs(wcs, shape) == PolarisationFrame("circular")

        wcs.wcs.crpix[0] = 1.0
        wcs.wcs.crval[0] = -5.0
        wcs.wcs.cdelt[0] = -1.0
        assert polarisation_frame_from_wcs(wcs, shape) == PolarisationFrame("linear")

        wcs.wcs.crpix[0] = 1.0
        wcs.wcs.crval[0] = -1.0
        wcs.wcs.cdelt[0] = -1.0
        assert polarisation_frame_from_wcs(wcs, shape) == PolarisationFrame("circular")

        with self.assertRaises(ValueError):
            wcs.wcs.crpix[0] = 1.0
            wcs.wcs.crval[0] = -100.0
            wcs.wcs.cdelt[0] = -1.0
            polarisation_frame_from_wcs(wcs, shape)

    def test_polarisation_frame_from_wcs_jones(self):
        vp = import_image_from_fits(
            rascil_data_path("models/MID_FEKO_VP_B2_45_1360_real.fits"), fixpol=False
        )
        imag_vp = import_image_from_fits(
            rascil_data_path("models/MID_FEKO_VP_B2_45_1360_imag.fits"), fixpol=False
        )
        vp["pixels"].data = vp["pixels"].data + 1j * imag_vp["pixels"].data

        polframe = polarisation_frame_from_wcs(vp.image_acc.wcs, vp.image_acc.shape)
        permute = polframe.fits_to_rascil[polframe.type]

        newvp_data = vp["pixels"].data.copy()
        for ip, p in enumerate(permute):
            newvp_data[:, p, ...] = vp["pixels"].data[:, ip, ...]
        vp["pixels"].data = newvp_data

        assert vp.image_acc.polarisation_frame == PolarisationFrame("linear")

    def test_smooth_image(self):
        assert numpy.max(self.m31image["pixels"].data) > 0.0
        smooth = smooth_image(self.m31image)
        assert numpy.max(smooth["pixels"].data) > 0.0
        assert numpy.max(smooth["pixels"].data) > numpy.max(
            self.m31image["pixels"].data
        )

    def test_integrate_image_over_frequency(self):
        frequency = numpy.linspace(0.9e8, 1.1e8, 9)
        original_cube = create_low_test_image_from_gleam(
            npixel=512, cellsize=0.0001, frequency=frequency, flux_limit=1.0
        )
        average = average_image_over_frequency(original_cube)
        assert average["pixels"].data.shape[0] == 1
        assert (
            numpy.abs(
                numpy.sum(original_cube["pixels"].data) / 9
                - numpy.sum(average["pixels"].data)
            )
            < 1e-12
        )

    def test_calculate_image_frequency_moments(self):
        frequency = numpy.linspace(0.9e8, 1.1e8, 9)
        original_cube = create_low_test_image_from_gleam(
            npixel=512, cellsize=0.001, frequency=frequency, flux_limit=1.0
        )
        original_cube = smooth_image(original_cube, width=1.0)

        if self.persist:
            export_image_to_fits(
                original_cube,
                fitsfile="%s/test_moments_original_cube.fits" % (self.dir),
            )
        moment_cube = calculate_image_frequency_moments(original_cube, nmoment=3)
        if self.persist:
            export_image_to_fits(
                moment_cube, fitsfile="%s/test_moments_moment_cube.fits" % (self.dir)
            )
        reconstructed_cube = calculate_image_from_frequency_moments(
            original_cube, moment_cube
        )
        print(reconstructed_cube.image_acc.wcs)
        if self.persist:
            export_image_to_fits(
                reconstructed_cube,
                fitsfile="%s/test_moments_reconstructed_cube.fits" % (self.dir),
            )
        error = numpy.std(
            reconstructed_cube["pixels"].data - original_cube["pixels"].data
        )
        assert error < 3.4, error

    def test_calculate_image_taylor_terms(self):
        frequency = numpy.linspace(0.9e8, 1.1e8, 9)
        original_cube = create_low_test_image_from_gleam(
            npixel=512,
            cellsize=0.001,
            frequency=frequency,
            flux_limit=1.0,
            applybeam=True,
        )
        original_cube = smooth_image(original_cube, width=1.0)
        if self.persist:
            export_image_to_fits(
                original_cube,
                fitsfile="%s/test_taylor_terms_original.fits" % (self.dir),
            )
        moment_cube = calculate_image_frequency_moments(original_cube, nmoment=3)
        if self.persist:
            export_image_to_fits(
                moment_cube,
                fitsfile="%s/test_taylor_terms_moments_cube.fits" % (self.dir),
            )
        taylor_term_images = calculate_image_taylor_terms(
            moment_cube,
            original_cube,
        )
        if self.persist:
            for moment, moment_image in enumerate(taylor_term_images):
                export_image_to_fits(
                    moment_image,
                    fitsfile=f"{self.dir}/test_taylor_terms_term{moment}_plane.fits",
                )

    def test_create_w_term_image(self):
        phasecentre = SkyCoord(
            ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
        )
        newimage = create_image(
            npixel=1024,
            cellsize=0.001,
            polarisation_frame=PolarisationFrame("stokesIQUV"),
            frequency=numpy.linspace(0.8e9, 1.2e9, 5),
            phasecentre=phasecentre,
            channel_bandwidth=1e7 * numpy.ones([5]),
        )
        im = create_w_term_like(newimage, w=2000.0, remove_shift=True, dopol=True)
        im["pixels"].data = im["pixels"].data.real
        for x in [256, 768]:
            for y in [256, 768]:
                self.assertAlmostEqual(
                    im["pixels"].data[0, 0, y, x], -0.46042631800538464, 7
                )
        if self.persist:
            export_image_to_fits(im, "%s/test_wterm.fits" % self.dir)
        assert im["pixels"].data.shape == (5, 4, 1024, 1024), im["pixels"].data.shape
        self.assertAlmostEqual(numpy.max(im["pixels"].real), 1.0, 7)

    def test_fftim(self):
        m31_fft = fft_image_to_griddata(self.m31image)
        m31_fft_ifft = ifft_griddata_to_image(m31_fft, self.m31image)
        m31_fft_ifft["pixels"] = m31_fft_ifft["pixels"].real
        if self.persist:
            export_image_to_fits(
                m31_fft_ifft, fitsfile="%s/test_m31_fft_fft.fits" % (self.dir)
            )
        err = numpy.max(
            numpy.abs(self.m31image["pixels"].data - m31_fft_ifft["pixels"].data)
        )
        assert err < 1e-7, err
        m31_fft_ifft["pixels"].data = numpy.abs(m31_fft_ifft["pixels"].data)

    def test_fftim_factors(self):
        for i in [3, 5, 7]:
            npixel = 256 * i
            m31image = create_test_image(cellsize=0.001, frequency=[1e8])
            padded = pad_image(m31image, [1, 1, npixel, npixel])
            assert padded.image_acc.shape == (1, 1, npixel, npixel)
            padded_fft = fft_image_to_griddata(padded)
            padded_fft_ifft = ifft_griddata_to_image(padded_fft, m31image)
            err = numpy.max(
                numpy.abs(padded["pixels"].data - padded_fft_ifft["pixels"].data.real)
            )
            assert err < 1e-7, err
            padded_fft["pixels"].data = numpy.abs(padded_fft["pixels"].data)
            if self.persist:
                export_image_to_fits(
                    padded_fft, fitsfile="%s/test_m31_fft_%d.fits" % (self.dir, npixel)
                )

    def test_pad_image(self):
        m31image = create_test_image(cellsize=0.001, frequency=[1e8])
        padded = pad_image(m31image, [1, 1, 1024, 1024])
        assert padded.image_acc.shape == (1, 1, 1024, 1024)

        with self.assertRaises(ValueError):
            padded = pad_image(m31image, [1, 1, 100, 100])

        with self.assertRaises(IndexError):
            padded = pad_image(m31image, [1, 1])

    def test_sub_image(self):
        m31image = create_test_image(cellsize=0.001, frequency=[1e8])
        sub = sub_image(m31image, [1, 1, 64, 64])
        assert sub.image_acc.shape == (1, 1, 64, 64)

    def test_scale_and_rotate(self):

        vp = create_vp(telescope="MID_FEKO_B2")
        vp = scale_and_rotate_image(vp, 90.0 * numpy.pi / 180.0)
        if self.persist:
            vp["pixels"].data = vp["pixels"].data.real
            fitsfile = "{}/test_vp_rotate_real.fits".format(self.dir)
            export_image_to_fits(vp, fitsfile=fitsfile)

    def test_apply_voltage_pattern(self):

        vp = create_vp(telescope="MID_FEKO_B2")
        cellsize = vp.image_acc.wcs.wcs.cdelt[1] * numpy.pi / 180.0
        m31image = create_test_image(
            cellsize=cellsize,
            frequency=[1.36e9],
            polarisation_frame=PolarisationFrame("stokesIQUV"),
        )
        padded = pad_image(m31image, vp["pixels"].shape)
        padded["pixels"].data[:, 1:, ...] = 0.0
        applied = apply_voltage_pattern_to_image(padded, vp)
        unapplied = apply_voltage_pattern_to_image(
            applied, vp, inverse=True, min_det=1e-12
        )
        if self.persist:
            applied["pixels"].data = applied["pixels"].data.real
            fitsfile = "{}/test_apply_voltage_pattern_real.fits".format(self.dir)
            export_image_to_fits(applied, fitsfile=fitsfile)
            unapplied["pixels"].data = unapplied["pixels"].data.real
            fitsfile = "{}/test_apply_voltage_pattern_inv_real.fits".format(self.dir)
            export_image_to_fits(unapplied, fitsfile=fitsfile)

        err = numpy.max(numpy.abs(unapplied["pixels"].data - padded["pixels"].data))
        assert err < 1e-12, err


if __name__ == "__main__":
    unittest.main()
