""" Image functions using taylor terms in frequency

"""

__all__ = [
    "calculate_frequency_taylor_terms_from_skycomponents",
]
import copy
import logging
from typing import List

import numpy

from rascil.data_models import Skycomponent
from rascil.processing_components import copy_skycomponent

log = logging.getLogger("rascil-logger")


def calculate_frequency_taylor_terms_from_skycomponents(
    sc_list: List[Skycomponent], nmoment=1, reference_frequency=None
) -> List[Skycomponent]:
    """Calculate frequency taylor terms

    .. math::

        w_k = \\left(\\left(\\nu - \\nu_{ref}\\right) /  \\nu_{ref}\\right)^k

    :param im_list: Image list to be reconstructed
    :param moment_image: Moment cube (constructed using calculate_image_frequency_moments)
    :param reference_frequency: Reference frequency (default None uses average)
    :return: list of reconstructed images
    """
    nchan = len(sc_list)
    single_chan = len(sc_list[0].frequency)
    if single_chan > 1:
        raise ValueError(
            "calculate_frequency_taylor_terms_from_skycomponent_list: each skycomponent list must be single channel"
        )

    frequency = numpy.array([sc.frequency.data[0] for sc in sc_list])

    if reference_frequency is None:
        reference_frequency = frequency[len(frequency) // 2]
    log.debug(
        "calculate_image_from_frequency_moments: Reference frequency = %.3f (MHz)"
        % (1e-6 * reference_frequency)
    )

    polarisation_frame = sc_list[0].polarisation_frame

    channel_moment_coupling = numpy.zeros([nchan, nmoment])
    for chan in range(nchan):
        for m in range(nmoment):
            channel_moment_coupling[chan, m] += numpy.power(
                (frequency[chan] - reference_frequency) / reference_frequency,
                m,
            )

    pinv = numpy.linalg.pinv(channel_moment_coupling, rcond=1e-7)

    decoupled_sc_list = list()
    for moment in range(nmoment):
        decoupled_data = numpy.zeros([1, polarisation_frame.npol])
        for chan in range(nchan):
            decoupled_data[0] += pinv[moment, chan] * sc_list[chan].flux[0]
        decoupled_sc = copy_skycomponent(sc_list[0])
        decoupled_sc.flux = decoupled_data
        decoupled_sc.frequency = reference_frequency
        decoupled_sc_list.append(decoupled_sc)

    return decoupled_sc_list
