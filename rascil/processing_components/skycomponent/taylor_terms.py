""" Image functions using taylor terms in frequency

"""

__all__ = [
    "calculate_frequency_taylor_terms_from_skycomponents",
    "find_skycomponents_frequency_taylor_terms",
    "interpolate_skycomponents_frequency",
    "transpose_skycomponents_to_channels",
    "gather_skycomponents_from_channels",
]
import copy
import logging
from typing import List

import numpy

from numpy.polynomial import polynomial as P

from rascil.data_models import Skycomponent, Image, get_parameter
from rascil.processing_components import (
    copy_skycomponent,
    find_skycomponents,
    fit_skycomponent,
)
from rascil.processing_components.image.taylor_terms import (
    calculate_frequency_taylor_terms_from_image_list,
)

log = logging.getLogger("rascil-logger")


def calculate_frequency_taylor_terms_from_skycomponents(
    sc_list: List[Skycomponent], nmoment=1, reference_frequency=None
) -> List[List[Skycomponent]]:
    """Calculate frequency taylor terms for a list of skycomponents

    :param sc_list: Skycomponent
    :param reference_frequency: Reference frequency (default None uses centre point)
    :return: Skycomponents as one component per Taylor term
    """
    if len(sc_list) == 0:
        return [nmoment * list()]
    nchan = len(sc_list[0].frequency)
    if reference_frequency is None:
        reference_frequency = sc_list[0].frequency[len(sc_list[0].frequency) // 2]
    log.debug(
        "calculate_image_from_frequency_moments: Reference frequency = %.3f (MHz)"
        % (1e-6 * reference_frequency)
    )

    channel_moment_coupling = numpy.zeros([nchan, nmoment])
    for chan in range(nchan):
        for m in range(nmoment):
            channel_moment_coupling[chan, m] += numpy.power(
                (sc_list[0].frequency[chan] - reference_frequency)
                / reference_frequency,
                m,
            )

    pinv = numpy.linalg.pinv(channel_moment_coupling, rcond=1e-7)

    newsc_list = list()
    for sc in sc_list:
        taylor_term_sc_list = list()
        for moment in range(nmoment):
            taylor_term_data = numpy.zeros([1, sc.polarisation_frame.npol])
            for chan in range(nchan):
                taylor_term_data[0] += pinv[moment, chan] * sc.flux[chan, 0]
            taylor_term_sc = copy_skycomponent(sc)
            taylor_term_sc.flux = taylor_term_data
            taylor_term_sc.frequency = reference_frequency
            taylor_term_sc_list.append(taylor_term_sc)
        newsc_list.append(taylor_term_sc_list)

    return newsc_list


def find_skycomponents_frequency_taylor_terms(
    dirty_list: List[Image], nmoment=1, reference_frequency=None, **kwargs
) -> List[List[Skycomponent]]:
    """Find skycomponents by fitting to moment0, fit polynomial in frequency, return in frequency space

    .. math::

        w_k = \\left(\\left(\\nu - \\nu_{ref}\\right) /  \\nu_{ref}\\right)^k

    :param dirty_list: List of images to be searched. These should be different frequencies
    :param nmoment: Number of moments to be fitted
    :param reference_frequency: Reference frequency (default None uses centre frequency)
    :return: list of skycomponents
    """
    frequency = numpy.array([d.frequency[0] for d in dirty_list])

    if reference_frequency is None:
        reference_frequency = frequency[len(frequency) // 2]
    log.debug(
        "calculate_image_from_frequency_moments: Reference frequency = %.3f (MHz)"
        % (1e-6 * reference_frequency)
    )

    moment0_list = calculate_frequency_taylor_terms_from_image_list(
        dirty_list, nmoment=1, reference_frequency=reference_frequency
    )
    threshold = get_parameter(kwargs, "component_threshold", numpy.inf)
    try:
        moment0_skycomponents = find_skycomponents(moment0_list[0], threshold=threshold)
    except AssertionError:
        log.info(
            f"find_skycomponents_frequency_taylor_terms: No skycomponents found in moment 0"
        )
        return []

    ncomps = len(moment0_skycomponents)
    if ncomps > 0:
        log.info(
            f"find_skycomponents_frequency_taylor_terms: found {ncomps} skycomponents in moment 0"
        )
    else:
        return []

    found_component_list = []
    for isc, sc in enumerate(moment0_skycomponents):
        found_component = copy_skycomponent(sc)
        found_component.frequency = frequency
        found_component.flux = numpy.array(
            [list(fit_skycomponent(d, sc, **kwargs).flux[0, :]) for d in dirty_list]
        )
        found_component_list.append(found_component)
        log.info(f"Component {isc}: {found_component}")

    interpolated_sc_list = interpolate_skycomponents_frequency(
        found_component_list, nmoment=nmoment, reference_frequency=reference_frequency
    )
    return transpose_skycomponents_to_channels(interpolated_sc_list)


def interpolate_skycomponents_frequency(
    sc_list, nmoment=1, reference_frequency=None, **kwargs
) -> List[Skycomponent]:
    """Smooth skycomponent fluxes by fitting polynomial in frequency

    First the sources are found using the moment0 image, and then the
    flux is fit in frequency by a Taylor series with nmoment terms.
    The final result is a list of lists where the outer coordinate
    is frequency

    .. math::

        w_k = \\left(\\left(\\nu - \\nu_{ref}\\right) /  \\nu_{ref}\\right)^k

    :param im_list: Image list to be reconstructed
    :param moment_image: Moment cube (constructed using calculate_image_frequency_moments)
    :param reference_frequency: Reference frequency (default None uses average)
    :return: list of reconstructed images
    """
    frequency = sc_list[0].frequency

    if reference_frequency is None:
        reference_frequency = frequency[len(frequency) // 2]
    log.debug(
        "calculate_image_from_frequency_moments: Reference frequency = %.3f (MHz)"
        % (1e-6 * reference_frequency)
    )

    # Now fit in frequency and keep the model
    newsc_list = list()
    for sc in sc_list:
        newsc = copy_skycomponent(sc)
        x = (frequency - reference_frequency) / reference_frequency
        y = sc.flux
        coeffs = P.polyfit(x, y, nmoment - 1)
        newsc.flux = P.polyval(x, coeffs).T
        newsc_list.append(newsc)

    return newsc_list


def transpose_skycomponents_to_channels(
    sc_list: List[Skycomponent],
) -> List[List[Skycomponent]]:
    """Tranpose a component list from [source,chan] to [chan,source]

    :param sc_list:
    :return: List[List[Skycomponent]]
    """
    newsc_list = list()
    nchan = len(sc_list[0].frequency)
    for chan in range(nchan):
        chan_sc_list = list()
        for comp in sc_list:
            newcomp = copy_skycomponent(comp)
            newcomp.frequency = numpy.array([comp.frequency[chan]])
            newcomp.flux = comp.flux[chan, :][numpy.newaxis, :]
            chan_sc_list.append(newcomp)
        newsc_list.append(chan_sc_list)
    return newsc_list


def gather_skycomponents_from_channels(
    sc_list: List[List[Skycomponent]],
) -> List[Skycomponent]:
    """Gather a component list from [chan][source] to [source]*skycomponent

    :param sc_list:
    :return: List[List[Skycomponent]]
    """
    nsource = len(sc_list[0])
    nchan = len(sc_list)
    newsc_list = list()
    for source in range(nsource):
        newcomp = copy_skycomponent(sc_list[0][source])
        flux = numpy.array([sc_list[chan][source].flux[0, :] for chan in range(nchan)])
        frequency = numpy.array(
            [sc_list[chan][source].frequency[0] for chan in range(nchan)]
        )
        newcomp.frequency = frequency
        newcomp.flux = flux
        newsc_list.append(newcomp)
    return newsc_list
