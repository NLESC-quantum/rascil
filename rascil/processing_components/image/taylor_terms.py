""" Image functions using taylor terms in frequency

"""

__all__ = [
    "calculate_image_frequency_moments",
    "calculate_image_from_frequency_taylor_terms",
    "calculate_image_list_frequency_moments",
    "calculate_image_list_from_frequency_taylor_terms",
    "calculate_frequency_taylor_terms_from_image_list",
]
import copy
import logging
from typing import List

import numpy

from rascil.data_models import Image
from rascil.processing_components.image.operations import (
    image_is_canonical,
    create_image_from_array,
)

log = logging.getLogger("rascil-logger")


def calculate_image_frequency_moments(
    im: Image, reference_frequency=None, nmoment=1
) -> Image:
    """Calculate frequency weighted moments of an image cube

    The frequency moments are calculated using:

    .. math::

        w_k = \\left(\\left(\\nu - \\nu_{ref}\\right) /  \\nu_{ref}\\right)^k


    Note that the spectral axis is replaced by a MOMENT axis.

    For example, to find the moments and then reconstruct from just the moments::

        moment_cube = calculate_image_frequency_moments(model_multichannel, nmoment=5)

    :param im: Image cube
    :param reference_frequency: Reference frequency (default None uses average)
    :param nmoment: Number of moments to calculate
    :return: Moments image cube
    """
    assert image_is_canonical(im)

    assert nmoment > 0
    nchan, npol, ny, nx = im["pixels"].data.shape
    channels = numpy.arange(nchan)
    freq = im.image_acc.wcs.sub(["spectral"]).wcs_pix2world(channels, 0)[0]

    assert (
        nmoment <= nchan
    ), "Number of moments %d cannot exceed the number of channels %d" % (nmoment, nchan)

    if reference_frequency is None:
        reference_frequency = numpy.average(freq.data)
    log.debug(
        "calculate_image_frequency_moments: Reference frequency = %.3f (MHz)"
        % (reference_frequency / 1e6)
    )

    moment_data = numpy.zeros([nmoment, npol, ny, nx])

    assert not numpy.isnan(numpy.sum(im["pixels"].data)), "NaNs present in image data"

    for moment in range(nmoment):
        for chan in range(nchan):
            weight = numpy.power(
                (freq[chan] - reference_frequency) / reference_frequency, moment
            )
            moment_data[moment, ...] += im["pixels"].data[chan, ...] * weight

    assert not numpy.isnan(numpy.sum(moment_data)), "NaNs present in moment data"

    moment_wcs = copy.deepcopy(im.image_acc.wcs)

    moment_wcs.wcs.ctype[3] = "MOMENT"
    moment_wcs.wcs.crval[3] = 0.0
    moment_wcs.wcs.crpix[3] = 1.0
    moment_wcs.wcs.cdelt[3] = 1.0
    moment_wcs.wcs.cunit[3] = ""

    return create_image_from_array(
        moment_data, moment_wcs, im.image_acc.polarisation_frame
    )


def calculate_image_from_frequency_taylor_terms(
    im: Image, taylor_terms_image: Image, reference_frequency=None
) -> Image:
    """Calculate channel image from Taylor term expansion in frequency

    .. math::

        w_k = \\left(\\left(\\nu - \\nu_{ref}\\right) /  \\nu_{ref}\\right)^k

    Note that a new image is created.

    The Taylor term representation can be generated using MSMFS in deconvolve.

    :param im: Image cube to be reconstructed
    :param taylor_terms_image: Taylor terms cube
    :param reference_frequency: Reference frequency (default None uses average)
    :return: reconstructed image
    """
    # assert isinstance(im, Image)
    nchan, npol, ny, nx = im["pixels"].data.shape
    n_taylor_terms, mnpol, mny, mnx = taylor_terms_image["pixels"].data.shape
    assert n_taylor_terms > 0

    assert npol == mnpol
    assert ny == mny
    assert nx == mnx

    if reference_frequency is None:
        reference_frequency = numpy.average(im.frequency.data)
    log.debug(
        "calculate_image_from_frequency_moments: Reference frequency = %.3f (MHz)"
        % (1e-6 * reference_frequency)
    )

    newim_data = numpy.zeros_like(im["pixels"].data[...])
    for taylor_term in range(n_taylor_terms):
        for chan in range(nchan):
            weight = numpy.power(
                (im.frequency[chan].data - reference_frequency) / reference_frequency,
                taylor_term,
            )
            newim_data[chan, ...] += (
                taylor_terms_image["pixels"].data[taylor_term, ...] * weight
            )

    newim = create_image_from_array(
        newim_data,
        wcs=im.image_acc.wcs,
        polarisation_frame=im.image_acc.polarisation_frame,
    )
    return newim


def calculate_image_list_frequency_moments(
    im_list: List[Image], reference_frequency=None, nmoment=1
) -> Image:
    """Calculate frequency weighted moments of an image list

    The frequency moments are calculated using:

    .. math::

        w_k = \\left(\\left(\\nu - \\nu_{ref}\\right) /  \\nu_{ref}\\right)^k


    Note that the spectral axis is replaced by a MOMENT axis.

    For example, to find the moments and then reconstruct from just the moments::

        moment_cube = calculate_image_frequency_moments(model_multichannel, nmoment=5)
        reconstructed_cube = calculate_image_from_frequency_moments(model_multichannel, moment_cube)

    :param im_list: List of images
    :param reference_frequency: Reference frequency (default None uses average)
    :param nmoment: Number of moments to calculate
    :return: Moments image
    """

    if nmoment <= 0:
        raise ValueError(f"Number of moments {nmoment} must be > 0")

    if im_list[0] is None:
        return None

    nchan = len(im_list)
    _, npol, ny, nx = im_list[0]["pixels"].data.shape
    freq = [
        im.image_acc.wcs.sub(["spectral"]).wcs_pix2world([0], 0)[0] for im in im_list
    ]

    if nmoment > nchan:
        raise ValueError(
            "Number of moments %d cannot exceed the number of channels %d"
            % (nmoment, nchan)
        )

    if reference_frequency is None:
        reference_frequency = numpy.average(freq)
    log.debug(
        "calculate_image_frequency_moments: Reference frequency = %.3f (MHz)"
        % (reference_frequency / 1e6)
    )

    moment_data = numpy.zeros([nmoment, npol, ny, nx])

    for moment in range(nmoment):
        for chan, im in enumerate(im_list):
            weight = numpy.power(
                (freq[chan] - reference_frequency) / reference_frequency, moment
            )
            moment_data[moment, ...] += im["pixels"].data[0, ...] * weight

    moment_wcs = copy.deepcopy(im_list[0].image_acc.wcs)

    moment_wcs.wcs.ctype[3] = "MOMENT"
    moment_wcs.wcs.crval[3] = 0.0
    moment_wcs.wcs.crpix[3] = 1.0
    moment_wcs.wcs.cdelt[3] = 1.0
    moment_wcs.wcs.cunit[3] = ""

    return create_image_from_array(
        moment_data, moment_wcs, im_list[0].image_acc.polarisation_frame
    )


def calculate_image_list_from_frequency_taylor_terms(
    im_list: List[Image], moment_image: Image, reference_frequency=None
) -> List[Image]:
    """Calculate image list from frequency weighted moments

    .. math::

        w_k = \\left(\\left(\\nu - \\nu_{ref}\\right) /  \\nu_{ref}\\right)^k

    :param im: Image cube to be reconstructed
    :param moment_image: Moment cube (constructed using calculate_image_frequency_moments)
    :param reference_frequency: Reference frequency (default None uses average)
    :return: list of reconstructed images
    """
    # assert isinstance(im, Image)
    nchan = len(im_list)
    nmoment, mnpol, mny, mnx = moment_image["pixels"].data.shape
    assert nmoment > 0

    frequency = numpy.array([d.frequency.data[0] for d in im_list])

    if reference_frequency is None:
        reference_frequency = numpy.average(frequency)
    log.debug(
        "calculate_image_from_frequency_moments: Reference frequency = %.3f (MHz)"
        % (1e-6 * reference_frequency)
    )

    newims = list()
    for chan in range(nchan):
        newim_data = numpy.zeros_like(im_list[chan]["pixels"].data[...])
        for moment in range(nmoment):
            weight = numpy.power(
                (frequency[chan] - reference_frequency) / reference_frequency,
                moment,
            )
            newim_data[0, ...] += moment_image["pixels"].data[moment, ...] * weight

        newim = create_image_from_array(
            newim_data,
            wcs=im_list[chan].image_acc.wcs,
            polarisation_frame=im_list[chan].image_acc.polarisation_frame,
        )

        newims.append(newim)

    return newims


def calculate_frequency_taylor_terms_from_image_list(
    im_list: List[Image], nmoment=1, reference_frequency=None
) -> List[Image]:
    """Calculate frequency taylor terms

    .. math::

        w_k = \\left(\\left(\\nu - \\nu_{ref}\\right) /  \\nu_{ref}\\right)^k

    :param im_list: Image list to be reconstructed
    :param moment_image: Moment cube (constructed using calculate_image_frequency_moments)
    :param reference_frequency: Reference frequency (default None uses average)
    :return: list of reconstructed images
    """
    # assert isinstance(im, Image)
    nchan = len(im_list)
    single_chan, npol, ny, nx = im_list[0]["pixels"].shape
    assert single_chan == 1

    frequency = numpy.array([d.frequency.data[0] for d in im_list])

    if reference_frequency is None:
        reference_frequency = frequency[len(frequency) // 2]
    log.debug(
        "calculate_image_from_frequency_moments: Reference frequency = %.3f (MHz)"
        % (1e-6 * reference_frequency)
    )

    wcs = im_list[nchan // 2].image_acc.wcs.deepcopy()
    wcs.wcs.ctype[3] = "FREQ"
    wcs.wcs.crval[3] = reference_frequency
    wcs.wcs.crpix[3] = 1.0
    wcs.wcs.cdelt[3] = 1.0
    wcs.wcs.cunit[3] = "Hz"

    polarisation_frame = im_list[nchan // 2].image_acc.polarisation_frame

    channel_moment_coupling = numpy.zeros([nchan, nmoment])
    for chan in range(nchan):
        for m in range(nmoment):
            channel_moment_coupling[chan, m] += numpy.power(
                (frequency[chan] - reference_frequency) / reference_frequency,
                m,
            )

    pinv = numpy.linalg.pinv(channel_moment_coupling, rcond=1e-7)

    decoupled_images = list()
    for moment in range(nmoment):
        decoupled_data = numpy.zeros([1, npol, ny, nx])
        for chan in range(nchan):
            decoupled_data[0] += pinv[moment, chan] * im_list[chan]["pixels"][0, ...]
        decoupled_image = Image(
            data=decoupled_data,
            wcs=wcs,
            polarisation_frame=polarisation_frame,
        )
        decoupled_images.append(decoupled_image)

    return decoupled_images
