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
    "deconvolve_list",
    "restore_list",
    "deconvolve_cube",
    "restore_cube",
    "fit_psf",
]

from typing import List

import logging
import warnings

from astropy.modeling import models, fitting

import numpy
from astropy.convolution import Gaussian2DKernel, convolve_fft

from rascil.processing_components.image.operations import (
    convert_clean_beam_to_degrees,
    convert_clean_beam_to_pixels,
)
from rascil.data_models.memory_data_models import Image
from rascil.data_models.parameters import get_parameter
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.arrays.cleaners import (
    hogbom,
    hogbom_complex,
    msclean,
    msmfsclean,
)
from rascil.processing_components.image.taylor_terms import (
    calculate_image_list_frequency_moments,
    calculate_image_list_from_frequency_taylor_terms,
)
from rascil.processing_components.image.operations import create_image_from_array
from rascil.processing_components.image.gather_scatter import (
    image_gather_channels,
    image_scatter_channels,
)


log = logging.getLogger("rascil-logger")


def deconvolve_list(
    dirty_list: List[Image],
    psf_list: List[Image],
    sensitivity_list: List[Image] = None,
    prefix="",
    **kwargs,
) -> (List[Image], List[Image]):
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

        comp, residual = deconvolve_list(dirty_list, psf_list, niter=1000, gain=0.7, algorithm='msclean',
                                         scales=[0, 3, 10, 30], threshold=0.01)

    For the MFS clean, the psf must have number of channels >= 2 * nmoment

    :param dirty_list: list of dirty image
    :param psf_list: list of point spread function
    :param sensitivity_list: List of Sensitivity image (i.e. inverse noise level)
    :param prefix: Informational message for logging
    :param window_shape: Window description
    :param mask: Window in the form of an image, overrides window_shape
    :param algorithm: Cleaning algorithm: 'msclean'|'hogbom'|'hogbom-complex'|'mfsmsclean'
    :param gain: loop gain (float) 0.7
    :param threshold: Clean threshold (0.0)
    :param fractional_threshold: Fractional threshold (0.01)
    :param scales: Scales (in pixels) for multiscale ([0, 3, 10, 30])
    :param nmoment: Number of frequency moments (default 3)
    :param findpeak: Method of finding peak in mfsclean: 'Algorithm1'|'ASKAPSoft'|'CASA'|'RASCIL', Default is RASCIL.
    :return: component image_list, residual image_list

    See also
        :py:func:`rascil.processing_components.arrays.cleaners.hogbom`
        :py:func:`rascil.processing_components.arrays.cleaners.hogbom_complex`
        :py:func:`rascil.processing_components.arrays.cleaners.msclean`
        :py:func:`rascil.processing_components.arrays.cleaners.msmfsclean`

    """

    window_shape = get_parameter(kwargs, "window_shape", None)
    window_list = find_window_list(dirty_list, prefix, window_shape=window_shape)

    check_psf_peak(psf_list)

    psf_list = bound_psf_list(dirty_list, prefix, psf_list, **kwargs)

    check_psf_peak(psf_list)

    algorithm = get_parameter(kwargs, "algorithm", "msclean")
    if algorithm == "msclean":
        comp_image_list, residual_image_list = msclean_kernel_list(
            dirty_list, prefix, psf_list, window_list, sensitivity_list, **kwargs
        )
    elif (
        algorithm == "msmfsclean" or algorithm == "mfsmsclean" or algorithm == "mmclean"
    ):
        comp_image_list, residual_image_list = mmclean_kernel_list(
            dirty_list, prefix, psf_list, window_list, sensitivity_list, **kwargs
        )
    elif algorithm == "hogbom":
        comp_image_list, residual_image_list = hogbom_kernel_list(
            dirty_list, prefix, psf_list, window_list, **kwargs
        )
    elif algorithm == "hogbom-complex":
        comp_image_list, residual_image_list = complex_hogbom_kernel_list(
            dirty_list, psf_list, window_list, **kwargs
        )
    else:
        raise ValueError(
            "deconvolve_cube %s: Unknown algorithm %s" % (prefix, algorithm)
        )

    log.info("deconvolve_cube %s: Deconvolution finished" % (prefix))

    return comp_image_list, residual_image_list


def check_psf_peak(psf_list):
    """Check that all PSFs in a list have unit peak

    :param psf_list: List of PSF images
    """
    for ipsf, psf in enumerate(psf_list):
        pmax = psf["pixels"].data.max()
        numpy.testing.assert_approx_equal(
            pmax,
            1.0,
            err_msg=f"check_psf_peak: PSF {ipsf} does not have unit peak {pmax}",
        )


def find_window_list(dirty_list, prefix, window_shape=None, **kwargs):
    """Find a clean window from a dirty image

    The values for window_shape are:
        "quarter" - Inner quarter of image
        "no_edge" - all but window_edge pixels around the perimeter
        mask - If an Image, use as the window (overrides other options)
        None - Entire image

    :param dirty_list: Image of the dirty image
    :param prefix: Informational prefix for log messages
    :param window_shape: Shape of window
    :param kwargs:
    :return: Numpy array
    """
    if window_shape is None:
        log.info("deconvolve_cube %s: Cleaning entire image" % prefix)
        return None

    windows = list()
    for channel, dirty in enumerate(dirty_list):
        if window_shape == "quarter":
            log.info("deconvolve_cube %s: window is inner quarter" % prefix)
            qx = dirty["pixels"].shape[3] // 4
            qy = dirty["pixels"].shape[2] // 4
            window_array = numpy.zeros_like(dirty["pixels"].data)
            window_array[..., (qy + 1) : 3 * qy, (qx + 1) : 3 * qx] = 1.0
            log.info(
                "deconvolve_cube %s: Cleaning inner quarter of each sky plane" % prefix
            )
        elif window_shape == "no_edge":
            edge = get_parameter(kwargs, "window_edge", 16)
            nx = dirty["pixels"].shape[3]
            ny = dirty["pixels"].shape[2]
            window_array = numpy.zeros_like(dirty["pixels"].data)
            window_array[..., (edge + 1) : (ny - edge), (edge + 1) : (nx - edge)] = 1.0
            log.info(
                "deconvolve_cube %s: Window omits %d-pixel edge of each sky plane"
                % (prefix, edge)
            )
        else:
            raise ValueError("Window shape %s is not recognized" % window_shape)

        mask = get_parameter(kwargs, "mask", None)
        if isinstance(mask, Image):
            if window_array is not None:
                log.warning(
                    "deconvolve_cube %s: Overriding window_shape with mask image"
                    % (prefix)
                )
                window_array = mask["pixels"].data
        if window_array is not None:
            window_image = create_image_from_array(
                window_array,
                dirty.image_acc.wcs,
                dirty.image_acc.polarisation_frame,
            )
        else:
            window_image = None

        windows.append(window_image)
    return windows


def bound_psf_list(dirty_list, prefix, psf_list, psf_support=None, **kwargs):
    """Calculate the PSF within a given support

    :param dirty_list: Dirty image, used for default sizes
    :param prefix: Informational prefix to log messages
    :param psf_list: Point Spread Function
    :param psf_support: The half width of a box centered on the psf centre
    :param kwargs:
    :return: psf: bounded point spread function (i.e. with smaller size in x and y)
    """
    psfs = list()
    for channel, dirty in enumerate(dirty_list):
        psf = psf_list[channel]
        if psf_support is None:
            psf_support = max(
                dirty["pixels"].shape[2] // 2, dirty["pixels"].shape[3] // 2
            )

        if (psf_support <= psf["pixels"].shape[2] // 2) and (
            (psf_support <= psf["pixels"].shape[3] // 2)
        ):
            centre = [psf["pixels"].shape[2] // 2, psf["pixels"].shape[3] // 2]
            psf = psf.isel(
                x=slice((centre[0] - psf_support), (centre[0] + psf_support)),
                y=slice((centre[1] - psf_support), (centre[1] + psf_support)),
            )
            log.debug(
                "deconvolve_cube %s: PSF support = +/- %d pixels"
                % (prefix, psf_support)
            )
            log.debug(
                "deconvolve_cube %s: PSF shape %s"
                % (prefix, str(psf["pixels"].data.shape))
            )
        else:
            log.info("Using entire psf for dconvolution")
        psfs.append(psf)
    return psfs


def complex_hogbom_kernel_list(
    dirty_list: List[Image], psf_list: List[Image], window_list: List[Image], **kwargs
):
    """Complex Hogbom CLEAN of stokesIQUV image, operating of lists of single frequency images

    :param dirty_list: Image dirty image
    :param psf_list: Image Point Spread Function
    :param window: Window array (Bool) - clean where True
    :param gain: loop gain (float) 0.q
    :param threshold: Clean threshold (0.0)
    :param fractional_threshold: Fractional threshold (0.01)
    :return: component image_list, residual image_list
    """

    log.info(
        "complex_hogbom_kernel_list: Starting Hogbom-complex clean of each channel separately"
    )

    fracthresh, gain, niter, thresh, scales = common_arguments(**kwargs)

    comp_images = list()
    residual_images = list()

    # Clean each dirty image in the list
    for channel, dirty in enumerate(dirty_list):
        psf = psf_list[channel]
        window = window_list[channel]
        comp_array = numpy.zeros(dirty["pixels"].data.shape)
        residual_array = numpy.zeros(dirty["pixels"].data.shape)
        for pol in range(dirty["pixels"].data.shape[1]):
            if pol == 0 or pol == 3:
                if psf["pixels"].data[0, pol, :, :].max():
                    log.info(
                        "complex_hogbom_kernel_list: Processing pol %d, channel %d"
                        % (pol, channel)
                    )
                    if window is None:
                        (
                            comp_array[channel, pol, :, :],
                            residual_array[channel, pol, :, :],
                        ) = hogbom(
                            dirty["pixels"].data[0, pol, :, :],
                            psf["pixels"].data[0, pol, :, :],
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
                            dirty["pixels"].data[0, pol, :, :],
                            psf["pixels"].data[0, pol, :, :],
                            window["pixels"].data[0, pol, :, :],
                            gain,
                            thresh,
                            niter,
                            fracthresh,
                        )
                else:
                    log.info(
                        "complex_hogbom_kernel_list: Skipping pol %d, channel %d"
                        % (pol, channel)
                    )
            if pol == 1:
                if psf["pixels"].data[0, 1:2, :, :].max():
                    log.info(
                        "complex_hogbom_kernel_list: Processing pol 1 and 2, channel %d"
                        % (channel)
                    )
                    if window is None:
                        (
                            comp_array[channel, 1, :, :],
                            comp_array[channel, 2, :, :],
                            residual_array[channel, 1, :, :],
                            residual_array[channel, 2, :, :],
                        ) = hogbom_complex(
                            dirty["pixels"].data[0, 1, :, :],
                            dirty["pixels"].data[0, 2, :, :],
                            psf["pixels"].data[0, 1, :, :],
                            psf["pixels"].data[0, 2, :, :],
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
                            dirty["pixels"].data[0, 1, :, :],
                            dirty["pixels"].data[0, 2, :, :],
                            psf["pixels"].data[0, 1, :, :],
                            psf["pixels"].data[0, 2, :, :],
                            window["pixels"].data[0, pol, :, :],
                            gain,
                            thresh,
                            niter,
                            fracthresh,
                        )
                else:
                    log.info(
                        "complex_hogbom_kernel_list: Skipping pol 1 and 2, channel %d"
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
        comp_images.append(comp_image)
        residual_images.append(residual_image)

    return comp_images, residual_images


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


def hogbom_kernel_list(
    dirty_list: List[Image],
    prefix,
    psf_list: List[Image],
    window_list: List[Image],
    **kwargs,
):
    """Hogbom Clean, operating of lists of single frequency images

     See: Hogbom CLEAN A&A Suppl, 15, 417, (1974)

    :param dirty_list: List of dirty images
     :param prefix: Informational string to be used in log messages e.g. "cycle 1, subimage 42"
     :param psf_list: List of Point Spread Function
     :param window_list: List of window images
     :param sensitivity_list: List of sensitivity images
     :param gain: loop gain (float) 0.1
     :param threshold: Clean threshold (0.0)
     :param fractional_threshold: Fractional threshold (0.01)
     :param scales: Scales (in pixels) for multiscale ([0, 3, 10, 30])
     :param nmoment: Number of frequency moments (default 3)
     :param findpeak: Method of finding peak in mfsclean: 'Algorithm1'|'ASKAPSoft'|'CASA'|'RASCIL', Default is RASCIL.

     :return: component image_list, residual image_list
    """

    log.info(
        "hogbom_kernel_list %s: Starting Hogbom clean of each polarisation and channel separately"
        % prefix
    )

    fracthresh, gain, niter, thresh, scales = common_arguments(**kwargs)

    comp_images = []
    residual_images = []

    for channel, dirty in enumerate(dirty_list):
        psf = psf_list[channel]
        comp_array = numpy.zeros(dirty["pixels"].data.shape)
        residual_array = numpy.zeros(dirty["pixels"].data.shape)
        for pol in range(dirty["pixels"].data.shape[1]):
            if psf["pixels"].data[0, pol, :, :].max():
                log.info(
                    "hogbom_kernel_list %s: Processing pol %d, channel %d"
                    % (prefix, pol, channel)
                )
                if window_list is None or window_list[channel] is None:
                    (comp_array[:, pol, :, :], residual_array[:, pol, :, :],) = hogbom(
                        dirty["pixels"].data[0, pol, :, :],
                        psf["pixels"].data[0, pol, :, :],
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
                        dirty["pixels"].data[0, pol, :, :],
                        psf["pixels"].data[0, pol, :, :],
                        window_list[channel]["pixels"].data[channel, pol, :, :],
                        gain,
                        thresh,
                        niter,
                        fracthresh,
                        prefix,
                    )
            else:
                log.info(
                    "hogbom_kernel_list %s: Skipping pol %d, channel %d"
                    % (prefix, pol, channel)
                )
        comp_image = create_image_from_array(
            comp_array, dirty.image_acc.wcs, dirty.image_acc.polarisation_frame
        )
        residual_image = create_image_from_array(
            residual_array, dirty.image_acc.wcs, dirty.image_acc.polarisation_frame
        )
        comp_images.append(comp_image)
        residual_images.append(residual_image)

    return comp_images, residual_images


def mmclean_kernel_list(
    dirty_list: List[Image],
    prefix,
    psf_list: List[Image],
    window_list: List[Image] = None,
    sensitivity_list: List[Image] = None,
    **kwargs,
):
    """mfsmsclean, msmfsclean, mmclean: MultiScale Multi-Frequency CLEAN

    See: U. Rau and T. J. Cornwell,
    “A multi-scale multi-frequency deconvolution algorithm for synthesis imaging in radio interferometry,” A&A 532,
    A71 (2011).

    For the MFS clean, the psf must have number of channels >= 2 * nmoment

    :param dirty_list: List of dirty images
    :param prefix: Informational string to be used in log messages e.g. "cycle 1, subimage 42"
    :param psf_list: List of Point Spread Function
    :param window_list: List of window images
    :param sensitivity_list: List of sensitivity images
    :return: component image_list, residual image_list

    :return: component image_list, residual image_list

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
        "mmclean_kernel_list %s: Starting Multi-scale multi-frequency clean of each polarisation separately"
        % prefix
    )
    nmoment = get_parameter(kwargs, "nmoment", 3)
    if not (nmoment >= 1):
        raise ValueError(
            "Number of frequency moments must be greater than or equal to one"
        )

    nchan = len(dirty_list)
    if not (nchan > 2 * (nmoment - 1)):
        raise ValueError(
            "Requires nchan %d > 2 * (nmoment %d - 1)" % (nchan, 2 * (nmoment - 1))
        )
    dirty_taylor = calculate_image_list_frequency_moments(dirty_list, nmoment=nmoment)

    if sensitivity_list is not None:
        sensitivity_taylor = calculate_image_list_frequency_moments(
            sensitivity_list, nmoment=nmoment
        )
        sensitivity_taylor["pixels"].data /= nchan

    else:
        sensitivity_taylor = None

    if window_list is not None:
        window_taylor = calculate_image_list_frequency_moments(
            window_list, nmoment=nmoment
        )
        window_taylor["pixels"].data /= nchan
    else:
        window_taylor = None

    if nmoment > 1:
        psf_taylor = calculate_image_list_frequency_moments(
            psf_list, nmoment=2 * nmoment
        )
    else:
        psf_taylor = calculate_image_list_frequency_moments(psf_list, nmoment=1)

    psf_peak = numpy.max(psf_taylor["pixels"].data)
    dirty_taylor["pixels"].data /= psf_peak
    psf_taylor["pixels"].data /= psf_peak
    log.info(
        "mmclean_kernel_list %s: Shape of Dirty moments image %s"
        % (prefix, str(dirty_taylor["pixels"].shape))
    )
    log.info(
        "mmclean_kernel_list %s: Shape of PSF moments image %s"
        % (prefix, str(psf_taylor["pixels"].shape))
    )

    fracthresh, gain, niter, thresh, scales = common_arguments(**kwargs)

    gain = get_parameter(kwargs, "gain", 0.7)
    if not (0.0 < gain < 2.0):
        raise ValueError("Loop gain must be between 0 and 2")

    comp_array = numpy.zeros(dirty_taylor["pixels"].data.shape)
    residual_array = numpy.zeros(dirty_taylor["pixels"].data.shape)
    for pol in range(dirty_taylor["pixels"].data.shape[1]):
        if sensitivity_taylor is not None:
            sens = sensitivity_taylor["pixels"].data[:, pol, :, :]
        else:
            sens = None
        # Always use the moment 0, Stokes I PSF
        if psf_taylor["pixels"].data[0, 0, :, :].max():
            log.info("mmclean_kernel_list %s: Processing pol %d" % (prefix, pol))
            if window_taylor is None:
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
                    % (prefix, int(numpy.sum(window_taylor["pixels"].data[0, pol])))
                )
                comp_array[:, pol, :, :], residual_array[:, pol, :, :] = msmfsclean(
                    dirty_taylor["pixels"].data[:, pol, :, :],
                    psf_taylor["pixels"].data[:, 0, :, :],
                    window_taylor["pixels"].data[0, pol, :, :],
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
    comp_taylor = create_image_from_array(
        comp_array,
        dirty_taylor.image_acc.wcs,
        dirty_taylor.image_acc.polarisation_frame,
    )
    residual_taylor = create_image_from_array(
        residual_array,
        dirty_taylor.image_acc.wcs,
        dirty_taylor.image_acc.polarisation_frame,
    )
    log.info(
        "mmclean_kernel_list %s: calculating spectral image lists from frequency moment images"
        % prefix
    )
    comp_list = calculate_image_list_from_frequency_taylor_terms(
        dirty_list, comp_taylor
    )
    residual_list = calculate_image_list_from_frequency_taylor_terms(
        dirty_list, residual_taylor
    )
    return comp_list, residual_list


def msclean_kernel_list(
    dirty_list: List[Image],
    prefix,
    psf_list: List[Image],
    window_list: List[Image],
    sensitivity_list=None,
    **kwargs,
):
    """MultiScale CLEAN, operating of lists of single frequency images

    See: Cornwell, T.J., Multiscale CLEAN (IEEE Journal of Selected Topics in Sig Proc,
    2008 vol. 2 pp. 793-801)

    The clean search is performed on the product of the sensitivity image (if supplied) and
    the residual image. This gives a way to bias against high noise.

    :param dirty_list: List of dirty images
    :param prefix: Informational string to be used in log messages e.g. "cycle 1, subimage 42"
    :param psf_list: List of Point Spread Function
    :param window_list: List of window images
    :param sensitivity_list: List of sensitivity images
    :return: component image_list, residual image_list

    The following optional arguments can be passed via kwargs:

    :param fractional_threshold: Fractional threshold (0.01)
    :param gain: loop gain (float) 0.7
    :param niter: Number of clean iterations (int) 100
    :param threshold: Clean threshold (0.0)
    :param scales: Scales (in pixels) for multiscale ([0, 3, 10, 30])
    """
    log.info(
        "msclean_kernel_list %s: Starting Multi-scale clean of each polarisation and channel separately"
        % prefix
    )

    fracthresh, gain, niter, thresh, scales = common_arguments(**kwargs)

    comp_images = []
    residual_images = []

    for channel, dirty in enumerate(dirty_list):
        psf = psf_list[channel]

        comp_array = numpy.zeros_like(dirty["pixels"].data)
        residual_array = numpy.zeros_like(dirty["pixels"].data)

        for pol in range(dirty["pixels"].data.shape[1]):
            if sensitivity_list is not None and sensitivity_list[channel] is not None:
                sens = sensitivity_list[channel]["pixels"].data[0, pol, :, :]
            else:
                sens = None
            if psf["pixels"].data[0, pol, :, :].max():
                log.info(
                    "msclean_kernel_list %s: Processing pol %d, channel %d"
                    % (prefix, pol, channel)
                )
                if window_list is None or window_list[channel] is None:
                    (comp_array[0, pol, :, :], residual_array[0, pol, :, :],) = msclean(
                        dirty["pixels"].data[0, pol, :, :],
                        psf["pixels"].data[0, pol, :, :],
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
                    (comp_array[0, pol, :, :], residual_array[0, pol, :, :],) = msclean(
                        dirty["pixels"].data[0, pol, :, :],
                        psf["pixels"].data[0, pol, :, :],
                        window_list[channel]["pixels"].data[0, pol, :, :],
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
                    "msclean_kernel_list %s: Skipping pol %d, channel %d"
                    % (prefix, pol, channel)
                )
        comp_image = create_image_from_array(
            comp_array, dirty.image_acc.wcs, dirty.image_acc.polarisation_frame
        )
        residual_image = create_image_from_array(
            residual_array, dirty.image_acc.wcs, dirty.image_acc.polarisation_frame
        )
        comp_images.append(comp_image)
        residual_images.append(residual_image)

    return comp_images, residual_images


def restore_list(
    model_list: List[Image],
    psf_list: List[Image] = None,
    residual_list: List[Image] = None,
    clean_beam=None,
):
    """Restore the model image to the residuals

    The clean beam can be specified as a dictionary with
    fields "bmaj", "bmin" (both in arcsec) "bpa" in degrees.

    :param model_list: Model image list (i.e. deconvolved)
    :param psf_list: Input PSF list (nchan)
    :param residual_list: Residual image
    :param clean_beam: Clean beam e.g. {"bmaj":0.1, "bmin":0.05, "bpa":-60.0}. Units are deg, deg, deg
    :return: restored_list

    """

    restored_list = []

    for channel, model in enumerate(model_list):

        if psf_list is not None:
            psf = psf_list[channel]
        else:
            psf = None

        restored = model.copy(deep=True)
        if residual_list is not None:
            residual = residual_list[channel]
        else:
            residual = None

        if clean_beam is None:
            if psf is not None:
                clean_beam = fit_psf(psf)
                log.info(
                    "restore_list: Using fitted clean beam (deg, deg, deg) = {}".format(
                        clean_beam
                    )
                )
            else:
                raise ValueError(
                    "restore_list: Either the psf or the clean_beam must be specified"
                )
        else:
            log.info(
                "restore_list: Using clean beam  (deg, deg, deg) = {}".format(
                    clean_beam
                )
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
                    boundary="wrap",
                )
        if residual is not None:
            restored["pixels"].data += residual["pixels"].data

        restored["pixels"].data = restored["pixels"].data.astype("float")

        restored.attrs["clean_beam"] = clean_beam

        restored_list.append(restored)

    return restored_list


def deconvolve_cube(
    dirty: Image, psf: Image, sensitivity: Image = None, prefix="", **kwargs
) -> (Image, Image):
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
    dirty_list = image_scatter_channels(dirty)
    psf_list = image_scatter_channels(psf)
    if sensitivity is not None:
        sensitivity_list = image_scatter_channels(sensitivity)
    else:
        sensitivity_list = None

    comp_list, residual_list = deconvolve_list(
        dirty_list, psf_list, sensitivity=sensitivity_list, prefix=prefix, **kwargs
    )
    comp = image_gather_channels(comp_list)
    residual = image_gather_channels(residual_list)
    return comp, residual


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
    model_list = image_scatter_channels(model)
    residual_list = image_scatter_channels(residual)
    psf_list = image_scatter_channels(psf)

    restored_list = restore_list(
        model_list=model_list,
        psf_list=psf_list,
        residual_list=residual_list,
        clean_beam=clean_beam,
    )

    return image_gather_channels(restored_list)
