""" Image deconvolution functions

The standard deconvolution algorithms are provided:

    hogbom: Hogbom CLEAN See: Hogbom CLEAN A&A Suppl, 15, 417, (1974)
    
    msclean: MultiScale CLEAN See: Cornwell, T.J., Multiscale CLEAN (IEEE Journal of Selected Topics in Sig Proc,
    2008 vol. 2 pp. 793-801)

    mfsmsclean: MultiScale Multi-Frequency See: U. Rau and T. J. Cornwell, “A multi-scale multi-frequency
    deconvolution algorithm for synthesis imaging in radio interferometry,” A&A 532, A71 (2011).

For example to make dirty image and PSF, deconvolve, and then restore::

    model = create_image_from_visibility(vt, cellsize=0.001, npixel=256)
    dirty, sumwt = invert_2d(vt, model)
    psf, sumwt = invert_2d(vt, model, dopsf=True)

    comp, residual = deconvolve_cube(dirty, psf, niter=1000, threshold=0.001, fracthresh=0.01, window_shape='quarter',
                                 gain=0.7, algorithm='msclean', scales=[0, 3, 10, 30])

    restored = restore_cube(comp, psf, residual)
    
All functions return an image holding clean components and residual image

"""

__all__ = [
    "deconvolve_cube",
    "restore_cube",
    "fit_psf",
    "convert_clean_beam_to_pixels",
    "convert_clean_beam_to_degrees",
]

import logging
import warnings

import numpy
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.modeling import models, fitting

from rascil.data_models.memory_data_models import Image
from rascil.data_models.parameters import get_parameter
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.arrays.cleaners import (
    hogbom,
    hogbom_complex,
    msclean,
    msmfsclean,
)
from rascil.processing_components.image.operations import (
    calculate_image_frequency_moments,
    calculate_image_from_frequency_moments,
    image_is_canonical,
)
from rascil.processing_components.image.operations import create_image_from_array

# from photutils import fit_2dgaussian

# warnings.simplefilter('ignore', AstropyDeprecationWarning)

log = logging.getLogger("rascil-logger")


def deconvolve_cube(dirty: Image, psf: Image, sensitivity: Image = None,
                    prefix="", **kwargs) -> (Image, Image):
    """Clean using a variety of algorithms

    The algorithms available are:

    hogbom: Hogbom CLEAN See: Hogbom CLEAN A&A Suppl, 15, 417, (1974)

    hogbom-complex: Complex Hogbom CLEAN of stokesIQUV image

    msclean: MultiScale CLEAN See: Cornwell, T.J., Multiscale CLEAN (IEEE Journal of Selected Topics in Sig Proc,
    2008 vol. 2 pp. 793-801)

    mfsmsclean, msmfsclean, mmclean: MultiScale Multi-Frequency See: U. Rau and T. J. Cornwell,
    “A multi-scale multi-frequency deconvolution algorithm for synthesis imaging in radio interferometry,” A&A 532,
    A71 (2011).

    For example::

        comp, residual = deconvolve_cube(dirty, psf, niter=1000, gain=0.7, algorithm='msclean',
                                         scales=[0, 3, 10, 30], threshold=0.01)

    For the MFS clean, the psf must have number of channels >= 2 * nmoment

    :param dirty: Image dirty image
    :param psf: Image Point Spread Function
    :param sensitivity: Sensitivity image (i.e. inverse noise level)
    :param prefix: Informational message for logging
    :param window_shape: Window image (Bool) - clean where True
    :param mask: Window in the form of an image, overrides window_shape
    :param algorithm: Cleaning algorithm: 'msclean'|'hogbom'|'hogbom-complex'|'mfsmsclean'
    :param gain: loop gain (float) 0.7
    :param threshold: Clean threshold (0.0)
    :param fractional_threshold: Fractional threshold (0.01)
    :param scales: Scales (in pixels) for multiscale ([0, 3, 10, 30])
    :param nmoment: Number of frequency moments (default 3)
    :param findpeak: Method of finding peak in mfsclean: 'Algorithm1'|'ASKAPSoft'|'CASA'|'RASCIL', Default is RASCIL.
    :return: component image, residual image

    See also
        :py:func:`rascil.processing_components.arrays.cleaners.hogbom`
        :py:func:`rascil.processing_components.arrays.cleaners.hogbom_complex`
        :py:func:`rascil.processing_components.arrays.cleaners.msclean`
        :py:func:`rascil.processing_components.arrays.cleaners.msmfsclean`

    """

    window = find_window(dirty, prefix, **kwargs)

    psf = bound_psf(dirty, prefix, psf, **kwargs)

    algorithm = get_parameter(kwargs, "algorithm", "msclean")
    if algorithm == "msclean":
        comp_image, residual_image = msclean_kernel(
            dirty, prefix, psf, window, sensitivity, **kwargs
        )
    elif (
        algorithm == "msmfsclean" or algorithm == "mfsmsclean" or algorithm == "mmclean"
    ):
        comp_image, residual_image = mmclean_kernel(
            dirty, prefix, psf, window, sensitivity, **kwargs
        )
    elif algorithm == "hogbom":
        comp_image, residual_image = hogbom_kernel(dirty, prefix, psf, window, **kwargs)
    elif algorithm == "hogbom-complex":
        comp_image, residual_image = complex_hogbom_kernel(dirty, psf, window, **kwargs)
    else:
        raise ValueError(
            "deconvolve_cube %s: Unknown algorithm %s" % (prefix, algorithm)
        )

    log.info("deconvolve_cube %s: Deconvolution finished" % (prefix))

    return comp_image, residual_image


def find_window(dirty, prefix, window_shape=None, **kwargs):
    """Find a clean window from a dirty image

    The values for window_shape are:
        "quarter" - Inner quarter of image
        "no_edge" - all but window_edge pixels around the perimeter
        mask - If an Image, use as the window (overrides other options)
        None - Entire image

    :param dirty: Image of the dirty image
    :param prefix: Informational prefix for log messages
    :param window_shape: Shape of window
    :param kwargs:
    :return: Numpy array
    """
    if window_shape == "quarter":
        log.info("deconvolve_cube %s: window is inner quarter" % prefix)
        qx = dirty["pixels"].shape[3] // 4
        qy = dirty["pixels"].shape[2] // 4
        window = numpy.zeros_like(dirty["pixels"].data)
        window[..., (qy + 1) : 3 * qy, (qx + 1) : 3 * qx] = 1.0
        log.info(
            "deconvolve_cube %s: Cleaning inner quarter of each sky plane" % prefix
        )
    elif window_shape == "no_edge":
        edge = get_parameter(kwargs, "window_edge", 16)
        nx = dirty["pixels"].shape[3]
        ny = dirty["pixels"].shape[2]
        window = numpy.zeros_like(dirty["pixels"].data)
        window[..., (edge + 1) : (ny - edge), (edge + 1) : (nx - edge)] = 1.0
        log.info(
            "deconvolve_cube %s: Window omits %d-pixel edge of each sky plane"
            % (prefix, edge)
        )
    elif window_shape is None:
        log.info("deconvolve_cube %s: Cleaning entire image" % prefix)
        window = None
    else:
        raise ValueError("Window shape %s is not recognized" % window_shape)
    mask = get_parameter(kwargs, "mask", None)
    if isinstance(mask, Image):
        if window is not None:
            log.warning(
                "deconvolve_cube %s: Overriding window_shape with mask image" % (prefix)
            )
        window = mask["pixels"].data
    return window


def bound_psf(dirty, prefix, psf, psf_support=None, **kwargs):
    """Calculate the PSF within a given support

    :param dirty: Dirty image, used for default sizes
    :param prefix: Informational prefix to log messages
    :param psf: Point Spread Function
    :param psf_support: The half width of a box centered on the psf centre
    :param kwargs:
    :return: psf: bounded point spread function (i.e. with smaller size in x and y)
    """
    if psf_support is None:
        psf_support = max(dirty["pixels"].shape[2] // 2, dirty["pixels"].shape[3] // 2)

    if (psf_support <= psf["pixels"].shape[2] // 2) and (
        (psf_support <= psf["pixels"].shape[3] // 2)
    ):
        centre = [psf["pixels"].shape[2] // 2, psf["pixels"].shape[3] // 2]
        psf = psf.isel(
            x=slice((centre[0] - psf_support), (centre[0] + psf_support)),
            y=slice((centre[1] - psf_support), (centre[1] + psf_support)),
        )
        log.info(
            "deconvolve_cube %s: PSF support = +/- %d pixels" % (prefix, psf_support)
        )
        log.info(
            "deconvolve_cube %s: PSF shape %s" % (prefix, str(psf["pixels"].data.shape))
        )
    else:
        log.info("Using entire psf for dconvolution")
    return psf


def complex_hogbom_kernel(dirty, psf, window, **kwargs):
    """Complex Hogbom CLEAN of stokesIQUV image

    :param dirty: Image dirty image
    :param psf: Image Point Spread Function
    :param window: Window array (Bool) - clean where True
    :param gain: loop gain (float) 0.q
    :param threshold: Clean threshold (0.0)
    :param fractional_threshold: Fractional threshold (0.01)
    :return: component image, residual image
    """

    log.info(
        "deconvolve_cube_complex: Starting Hogbom-complex clean of each channel separately"
    )

    fracthresh, gain, niter, thresh, scales = common_arguments(**kwargs)

    comp_array = numpy.zeros(dirty["pixels"].data.shape)
    residual_array = numpy.zeros(dirty["pixels"].data.shape)
    for channel in range(dirty["pixels"].data.shape[0]):
        for pol in range(dirty["pixels"].data.shape[1]):
            if pol == 0 or pol == 3:
                if psf["pixels"].data[channel, pol, :, :].max():
                    log.info(
                        "deconvolve_cube_complex: Processing pol %d, channel %d"
                        % (pol, channel)
                    )
                    if window is None:
                        (
                            comp_array[channel, pol, :, :],
                            residual_array[channel, pol, :, :],
                        ) = hogbom(
                            dirty["pixels"].data[channel, pol, :, :],
                            psf["pixels"].data[channel, pol, :, :],
                            None,
                            gain,
                            thresh,
                            niter,
                            fracthresh,
                        )
                    else:
                        (
                            comp_array[channel, pol, :, :],
                            residual_array[channel, pol, :, :],
                        ) = hogbom(
                            dirty["pixels"].data[channel, pol, :, :],
                            psf["pixels"].data[channel, pol, :, :],
                            window[channel, pol, :, :],
                            gain,
                            thresh,
                            niter,
                            fracthresh,
                        )
                else:
                    log.info(
                        "deconvolve_cube_complex: Skipping pol %d, channel %d"
                        % (pol, channel)
                    )
            if pol == 1:
                if psf["pixels"].data[channel, 1:2, :, :].max():
                    log.info(
                        "deconvolve_cube_complex: Processing pol 1 and 2, channel %d"
                        % (channel)
                    )
                    if window is None:
                        (
                            comp_array[channel, 1, :, :],
                            comp_array[channel, 2, :, :],
                            residual_array[channel, 1, :, :],
                            residual_array[channel, 2, :, :],
                        ) = hogbom_complex(
                            dirty["pixels"].data[channel, 1, :, :],
                            dirty["pixels"].data[channel, 2, :, :],
                            psf["pixels"].data[channel, 1, :, :],
                            psf["pixels"].data[channel, 2, :, :],
                            None,
                            gain,
                            thresh,
                            niter,
                            fracthresh,
                        )
                    else:
                        (
                            comp_array[channel, 1, :, :],
                            comp_array[channel, 2, :, :],
                            residual_array[channel, 1, :, :],
                            residual_array[channel, 2, :, :],
                        ) = hogbom_complex(
                            dirty["pixels"].data[channel, 1, :, :],
                            dirty["pixels"].data[channel, 2, :, :],
                            psf["pixels"].data[channel, 1, :, :],
                            psf["pixels"].data[channel, 2, :, :],
                            window[channel, pol, :, :],
                            gain,
                            thresh,
                            niter,
                            fracthresh,
                        )
                else:
                    log.info(
                        "deconvolve_cube_complex: Skipping pol 1 and 2, channel %d"
                        % (channel)
                    )
            if pol == 2:
                continue
    comp_image = create_image_from_array(
        comp_array,
        dirty.image_acc.wcs,
        polarisation_frame=PolarisationFrame("stokesIQUV"),
    )
    residual_image = create_image_from_array(
        residual_array,
        dirty.image_acc.wcs,
        polarisation_frame=PolarisationFrame("stokesIQUV"),
    )
    return comp_image, residual_image


def common_arguments(**kwargs):
    """Extract the common arguments from kwargs

    :param gain: loop gain (float) default: 0.7
    :param niter: Number of minor cycle iterations: 100
    :param threshold: Clean threshold default 0.0
    :param fractional_threshold: Fractional threshold default 0.1
    :param scales: Scales (in pixels) for multiscale ([0, 3, 10, 30])

    :param kwargs:
    :return: fracthresh, gain, niter, thresh, scales
    """
    gain = get_parameter(kwargs, "gain", 0.1)
    if gain <= 0.0 or gain >= 2.0:
        raise ValueError("Loop gain must be between 0 and 2")
    thresh = get_parameter(kwargs, "threshold", 0.0)
    if thresh < 0.0:
        raise ValueError("Threshold must be positive or zero")
    niter = get_parameter(kwargs, "niter", 100)
    if niter < 0:
        raise ValueError("niter must be greater than zero")
    fracthresh = get_parameter(kwargs, "fractional_threshold", 0.01)
    if fracthresh < 0.0 or fracthresh > 1.0:
        raise ValueError("Fractional threshold should be in range 0.0, 1.0")
    scales = get_parameter(kwargs, "scales", [0, 3, 10, 30])

    return fracthresh, gain, niter, thresh, scales


def hogbom_kernel(dirty, prefix, psf, window, **kwargs):
    """Hogbom Clean

    See: Hogbom CLEAN A&A Suppl, 15, 417, (1974)

    :param dirty: Image dirty image
    :param prefix: Informational message for logging
    :param psf: Image Point Spread Function
    :param window: Window array (Bool) - clean where True
    :param gain: loop gain (float) 0.1
    :param threshold: Clean threshold (0.0)
    :param fractional_threshold: Fractional threshold (0.01)
    :param scales: Scales (in pixels) for multiscale ([0, 3, 10, 30])
    :param nmoment: Number of frequency moments (default 3)
    :param findpeak: Method of finding peak in mfsclean: 'Algorithm1'|'ASKAPSoft'|'CASA'|'RASCIL', Default is RASCIL.

    :return: component image, residual image
    """

    log.info(
        "deconvolve_cube %s: Starting Hogbom clean of each polarisation and channel separately"
        % prefix
    )

    fracthresh, gain, niter, thresh, scales = common_arguments(**kwargs)

    comp_array = numpy.zeros(dirty["pixels"].data.shape)
    residual_array = numpy.zeros(dirty["pixels"].data.shape)
    for channel in range(dirty["pixels"].data.shape[0]):
        for pol in range(dirty["pixels"].data.shape[1]):
            if psf["pixels"].data[channel, pol, :, :].max():
                log.info(
                    "deconvolve_cube %s: Processing pol %d, channel %d"
                    % (prefix, pol, channel)
                )
                if window is None:
                    (
                        comp_array[channel, pol, :, :],
                        residual_array[channel, pol, :, :],
                    ) = hogbom(
                        dirty["pixels"].data[channel, pol, :, :],
                        psf["pixels"].data[channel, pol, :, :],
                        None,
                        gain,
                        thresh,
                        niter,
                        fracthresh,
                        prefix,
                    )
                else:
                    (
                        comp_array[channel, pol, :, :],
                        residual_array[channel, pol, :, :],
                    ) = hogbom(
                        dirty["pixels"].data[channel, pol, :, :],
                        psf["pixels"].data[channel, pol, :, :],
                        window[channel, pol, :, :],
                        gain,
                        thresh,
                        niter,
                        fracthresh,
                        prefix,
                    )
            else:
                log.info(
                    "deconvolve_cube %s: Skipping pol %d, channel %d"
                    % (prefix, pol, channel)
                )
    comp_image = create_image_from_array(
        comp_array, dirty.image_acc.wcs, dirty.image_acc.polarisation_frame
    )
    residual_image = create_image_from_array(
        residual_array, dirty.image_acc.wcs, dirty.image_acc.polarisation_frame
    )
    return comp_image, residual_image


def mmclean_kernel(dirty, prefix, psf, window, sensitivity, **kwargs):
    """mfsmsclean, msmfsclean, mmclean: MultiScale Multi-Frequency CLEAN

    See: U. Rau and T. J. Cornwell,
    “A multi-scale multi-frequency deconvolution algorithm for synthesis imaging in radio interferometry,” A&A 532,
    A71 (2011).

    For the MFS clean, the psf must have number of channels >= 2 * nmoment

    :param dirty: Image dirty image
    :param prefix: Informational string to be used in log messages e.g. "cycle 1, subimage 42"
    :param psf: Image Point Spread Function
    :param window: Window image (Bool) - clean where True
    :param sensitivity: sensitivity image
    :return: component image, residual image

    The following optional arguments can be passed via kwargs:
    
    :param fractional_threshold: Fractional threshold (0.01)
    :param gain: loop gain (float) 0.7
    :param niter: Number of clean iterations (int) 100
    :param threshold: Clean threshold (0.0)
    :param scales: Scales (in pixels) for multiscale ([0, 3, 10, 30])
    :param nmoment: Number of frequency moments (default 3)
    :param findpeak: Method of finding peak in mfsclean: 'Algorithm1'|'CASA'|'RASCIL', Default is RASCIL.

    """

    findpeak = get_parameter(kwargs, "findpeak", "RASCIL")
    log.info(
        "deconvolve_cube %s: Starting Multi-scale multi-frequency clean of each polarisation separately"
        % prefix
    )
    nmoment = get_parameter(kwargs, "nmoment", 3)
    if not (nmoment >= 1):
        raise ValueError(
            "Number of frequency moments must be greater than or equal to one"
        )
    nchan = dirty["pixels"].shape[0]
    if not (nchan > 2 * (nmoment - 1)):
        raise ValueError(
            "Require nchan %d > 2 * (nmoment %d - 1)" % (nchan, 2 * (nmoment - 1))
        )
    dirty_taylor = calculate_image_frequency_moments(dirty, nmoment=nmoment)
    if nmoment > 1:
        psf_taylor = calculate_image_frequency_moments(psf, nmoment=2 * nmoment)
    else:
        psf_taylor = calculate_image_frequency_moments(psf, nmoment=1)
    psf_peak = numpy.max(psf_taylor["pixels"].data)
    dirty_taylor["pixels"].data /= psf_peak
    psf_taylor["pixels"].data /= psf_peak
    log.info(
        "deconvolve_cube %s: Shape of Dirty moments image %s"
        % (prefix, str(dirty_taylor["pixels"].shape))
    )
    log.info(
        "deconvolve_cube %s: Shape of PSF moments image %s"
        % (prefix, str(psf_taylor["pixels"].shape))
    )

    fracthresh, gain, niter, thresh, scales = common_arguments(**kwargs)

    gain = get_parameter(kwargs, "gain", 0.7)
    if not (0.0 < gain < 2.0):
        raise ValueError("Loop gain must be between 0 and 2")

    comp_array = numpy.zeros(dirty_taylor["pixels"].data.shape)
    residual_array = numpy.zeros(dirty_taylor["pixels"].data.shape)
    for pol in range(dirty_taylor["pixels"].data.shape[1]):
        if sensitivity is not None:
            sens = sensitivity["pixels"].data[:, pol, :, :]
        else:
            sens = None
        # Always use the Stokes I PSF
        if psf_taylor["pixels"].data[0, 0, :, :].max():
            log.info("deconvolve_cube %s: Processing pol %d" % (prefix, pol))
            if window is None:
                comp_array[:, pol, :, :], residual_array[:, pol, :, :] = msmfsclean(
                    dirty_taylor["pixels"].data[:, pol, :, :],
                    psf_taylor["pixels"].data[:, 0, :, :],
                    None,
                    sens,
                    gain,
                    thresh,
                    niter,
                    scales,
                    fracthresh,
                    findpeak,
                    prefix,
                )
            else:
                log.info(
                    "deconvolve_cube %s: Clean window has %d valid pixels"
                    % (prefix, int(numpy.sum(window[0, pol])))
                )
                comp_array[:, pol, :, :], residual_array[:, pol, :, :] = msmfsclean(
                    dirty_taylor["pixels"].data[:, pol, :, :],
                    psf_taylor["pixels"].data[:, 0, :, :],
                    window[0, pol, :, :],
                    sens,
                    gain,
                    thresh,
                    niter,
                    scales,
                    fracthresh,
                    findpeak,
                    prefix,
                )
        else:
            log.info("deconvolve_cube %s: Skipping pol %d" % (prefix, pol))
    comp_image = create_image_from_array(
        comp_array, dirty.image_acc.wcs, dirty.image_acc.polarisation_frame
    )
    residual_image = create_image_from_array(
        residual_array, dirty.image_acc.wcs, dirty.image_acc.polarisation_frame
    )
    log.info(
        "deconvolve_cube %s: calculating spectral cubes from frequency moment images"
        % prefix
    )
    comp_image = calculate_image_from_frequency_moments(dirty, comp_image)
    residual_image = calculate_image_from_frequency_moments(dirty, residual_image)
    return comp_image, residual_image


def msclean_kernel(dirty, prefix, psf, window, sensitivity=None, **kwargs):
    """MultiScale CLEAN

    See: Cornwell, T.J., Multiscale CLEAN (IEEE Journal of Selected Topics in Sig Proc,
    2008 vol. 2 pp. 793-801)
    
    The clean search is performed on the product of the sensitivity image (if supplied) and
    the residual image. This gives a way to bias against af high noise.

    :param dirty: Image dirty image
    :param prefix: Informational string to be used in log messages e.g. "cycle 1, subimage 42"
    :param psf: Image Point Spread Function
    :param window: Window image (Bool) - clean where True
    :param sensitivity: sensitivity image
    :return: component image, residual image

    The following optional arguments can be passed via kwargs:
    
    :param fractional_threshold: Fractional threshold (0.01)
    :param gain: loop gain (float) 0.7
    :param niter: Number of clean iterations (int) 100
    :param threshold: Clean threshold (0.0)
    :param scales: Scales (in pixels) for multiscale ([0, 3, 10, 30])
    """
    log.info(
        "deconvolve_cube %s: Starting Multi-scale clean of each polarisation and channel separately"
        % prefix
    )

    fracthresh, gain, niter, thresh, scales = common_arguments(**kwargs)

    comp_array = numpy.zeros_like(dirty["pixels"].data)
    residual_array = numpy.zeros_like(dirty["pixels"].data)
    for channel in range(dirty["pixels"].data.shape[0]):
        for pol in range(dirty["pixels"].data.shape[1]):
            if sensitivity is not None:
                sens = sensitivity["pixels"].data[channel, pol, :, :]
            else:
                sens = None
            if psf["pixels"].data[channel, pol, :, :].max():
                log.info(
                    "deconvolve_cube %s: Processing pol %d, channel %d"
                    % (prefix, pol, channel)
                )
                if window is None:
                    (
                        comp_array[channel, pol, :, :],
                        residual_array[channel, pol, :, :],
                    ) = msclean(
                        dirty["pixels"].data[channel, pol, :, :],
                        psf["pixels"].data[channel, pol, :, :],
                        None,
                        sens,
                        gain,
                        thresh,
                        niter,
                        scales,
                        fracthresh,
                        prefix,
                    )
                else:
                    (
                        comp_array[channel, pol, :, :],
                        residual_array[channel, pol, :, :],
                    ) = msclean(
                        dirty["pixels"].data[channel, pol, :, :],
                        psf["pixels"].data[channel, pol, :, :],
                        window[channel, pol, :, :],
                        sens,
                        gain,
                        thresh,
                        niter,
                        scales,
                        fracthresh,
                        prefix,
                    )
            else:
                log.info(
                    "deconvolve_cube %s: Skipping pol %d, channel %d"
                    % (prefix, pol, channel)
                )
    comp_image = create_image_from_array(
        comp_array, dirty.image_acc.wcs, dirty.image_acc.polarisation_frame
    )
    residual_image = create_image_from_array(
        residual_array, dirty.image_acc.wcs, dirty.image_acc.polarisation_frame
    )
    return comp_image, residual_image


def fit_psf(psf: Image, **kwargs):
    """Fit a two dimensional Gaussian to a PSF using astropy.modeling

    :params psf: Input PSF
    :return: bmaj (arcsec), bmin (arcsec), bpa (deg)
    """
    npixel = psf["pixels"].data.shape[3]
    sl = slice(npixel // 2 - 7, npixel // 2 + 8)
    y, x = numpy.mgrid[sl, sl]
    z = psf["pixels"].data[0, 0, sl, sl]

    # isotropic at the moment!
    from scipy.optimize import minpack

    try:
        p_init = models.Gaussian2D(
            amplitude=numpy.max(z), x_mean=numpy.mean(x), y_mean=numpy.mean(y)
        )
        fit_p = fitting.LevMarLSQFitter()
        with warnings.catch_warnings():
            # Ignore model linearity warning from the fitter
            warnings.simplefilter("ignore")
            fit = fit_p(p_init, x, y, z)
        if fit.x_stddev <= 0.0 or fit.y_stddev <= 0.0:
            log.warning("fit_psf: error in fitting to psf, using 1 pixel stddev")
            beam_pixels = (1.0, 1.0, 0.0)
        else:
            # Note that the order here is minor, major, pa
            beam_pixels = (fit.x_stddev.value, fit.y_stddev.value, fit.theta.value)
            # log.debug('fit_psf: fitted (pixels, pixels, rad) = {}'.format(beam_pixels))
    except minpack.error as err:
        log.warning("fit_psf: minpack error, using 1 pixel stddev")
        beam_pixels = (1.0, 1.0, 0.0)
    except ValueError as err:
        log.warning("fit_psf: warning in fit to psf, using 1 pixel stddev")
        beam_pixels = (1.0, 1.0, 0.0)

    return convert_clean_beam_to_degrees(psf, beam_pixels)


def convert_clean_beam_to_degrees(im, beam_pixels):
    """Convert clean beam in pixels to deg deg, deg

    :param im: Image
    :param beam_pixels:
    :return: dict e.g. {"bmaj":0.1, "bmin":0.05, "bpa":-60.0}. Units are deg, deg, deg
    """
    # cellsize in radians
    cellsize = numpy.deg2rad(im.image_acc.wcs.wcs.cdelt[1])
    to_mm = 4.0 * numpy.log(2.0)
    if beam_pixels[1] > beam_pixels[0]:
        clean_beam = {
            "bmaj": numpy.rad2deg(beam_pixels[1] * cellsize * to_mm),
            "bmin": numpy.rad2deg(beam_pixels[0] * cellsize * to_mm),
            "bpa": numpy.rad2deg(beam_pixels[2]),
        }
    else:
        clean_beam = {
            "bmaj": numpy.rad2deg(beam_pixels[0] * cellsize * to_mm),
            "bmin": numpy.rad2deg(beam_pixels[1] * cellsize * to_mm),
            "bpa": numpy.rad2deg(beam_pixels[2]) + 90.0,
        }
    return clean_beam


def restore_cube(model: Image, psf=None, residual=None, clean_beam=None) -> Image:
    """Restore the model image to the residuals

    The clean beam can be specified as a dictionary with
    fields "bmaj", "bmin" (both in arcsec) "bpa" in degrees.

    :param model: Model image (i.e. deconvolved)
    :param psf: Input PSF
    :param residual: Residual image
    :param clean_beam: Clean beam e.g. {"bmaj":0.1, "bmin":0.05, "bpa":-60.0}. Units are deg, deg, deg
    :return: restored image

    """
    restored = model.copy(deep=True)

    if clean_beam is None:
        if psf is not None:
            clean_beam = fit_psf(psf)
            log.info(
                "restore_cube: Using fitted clean beam (deg, deg, deg) = {}".format(
                    clean_beam
                )
            )
        else:
            raise ValueError(
                "restore_cube: Either the psf or the clean_beam must be specified"
            )
    else:
        log.info(
            "restore_cube: Using clean beam  (deg, deg, deg) = {}".format(clean_beam)
        )

    beam_pixels = convert_clean_beam_to_pixels(model, clean_beam)

    gk = Gaussian2DKernel(
        x_stddev=beam_pixels[0], y_stddev=beam_pixels[1], theta=beam_pixels[2]
    )
    # By convention, we normalise the peak not the integral so this is the volume of the Gaussian
    norm = 2.0 * numpy.pi * beam_pixels[0] * beam_pixels[1]
    # gk = Gaussian2DKernel(size)
    for chan in range(model["pixels"].shape[0]):
        for pol in range(model["pixels"].shape[1]):
            restored["pixels"].data[chan, pol, :, :] = norm * convolve_fft(
                model["pixels"].data[chan, pol, :, :],
                gk,
                normalize_kernel=False,
                allow_huge=True,
            )
    if residual is not None:
        restored["pixels"].data += residual["pixels"].data

    restored["pixels"].data = restored["pixels"].data.astype("float")

    restored.attrs["clean_beam"] = clean_beam

    return restored


def convert_clean_beam_to_pixels(model, clean_beam):
    """Convert clean beam to pixels

    :param model:
    :param clean_beam: e.g. {"bmaj":0.1, "bmin":0.05, "bpa":-60.0}. Units are deg, deg, deg
    :return:
    """
    to_mm = 4.0 * numpy.log(2.0)
    # Cellsize in radians
    cellsize = numpy.deg2rad(model.image_acc.wcs.wcs.cdelt[1])
    # Beam in pixels
    beam_pixels = (
        numpy.deg2rad(clean_beam["bmin"]) / (cellsize * to_mm),
        numpy.deg2rad(clean_beam["bmaj"]) / (cellsize * to_mm),
        numpy.deg2rad(clean_beam["bpa"]),
    )
    return beam_pixels
