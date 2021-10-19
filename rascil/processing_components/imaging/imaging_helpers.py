""" Functions to aid operations on imaging results

"""

import numpy
import logging

from rascil.processing_components import create_empty_image_like, normalise_sumwt
from rascil.processing_components.image.taylor_terms import (
    calculate_image_frequency_moments,
)

log = logging.getLogger("rascil-logger")


def sum_invert_results(image_list):
    """Sum a set of invert results with appropriate weighting

    :param image_list: List of [image, sum weights] pairs
    :return: image, sum of weights
    """
    if len(image_list) == 1:
        im = image_list[0][0].copy(deep=True)
        sumwt = image_list[0][1]
        return im, sumwt
    else:

        im = create_empty_image_like(image_list[0][0])
        sumwt = image_list[0][1].copy()
        sumwt *= 0.0

        for i, arg in enumerate(image_list):
            if arg is not None:
                im["pixels"].data += (
                    arg[1][..., numpy.newaxis, numpy.newaxis] * arg[0]["pixels"].data
                )
                sumwt += arg[1]

        im = normalise_sumwt(im, sumwt)

        return im, sumwt


def remove_sumwt(results):
    """Remove sumwt term in list of tuples (image, sumwt)

    :param results:
    :return: A list of just the dirty images
    """
    try:
        return [d[0] for d in results]
    except KeyError:
        return results


def sum_predict_results(results):
    """Sum a set of predict results of the same shape

    :param results: List of visibilities to be summed
    :return: summed visibility
    """
    sum_results = None
    for result in results:
        if result is not None:
            if sum_results is None:
                sum_results = result
            else:
                assert sum_results["vis"].data.shape == result["vis"].data.shape
                sum_results["vis"].data += result["vis"].data

    return sum_results


def threshold_list(
    imagelist, threshold, fractional_threshold, use_moment0=True, prefix=""
):
    """Find actual threshold for list of results, optionally using moment 0

    :param prefix: Prefix in log messages
    :param imagelist:
    :param threshold: Absolute threshold
    :param fractional_threshold: Fractional  threshold
    :param use_moment0: Use moment 0 for threshold
    :return:
    """
    peak = 0.0
    for i, result in enumerate(imagelist):
        if use_moment0:
            moments = calculate_image_frequency_moments(result)
            this_peak = numpy.max(
                numpy.abs(moments["pixels"].data[0, ...] / result["pixels"].shape[0])
            )
            peak = max(peak, this_peak)
            log.info(
                "threshold_list: using moment 0, sub_image %d, peak = %f,"
                % (i, this_peak)
            )
        else:
            ref_chan = result["pixels"].data.shape[0] // 2
            this_peak = numpy.max(numpy.abs(result["pixels"].data[ref_chan]))
            peak = max(peak, this_peak)
            log.info(
                "threshold_list: using refchan %d , sub_image %d, peak = %f,"
                % (ref_chan, i, this_peak)
            )

    actual = max(peak * fractional_threshold, threshold)

    if use_moment0:
        log.info(
            "threshold_list %s: Global peak in moment 0 = %.6f, sub-image clean threshold will be %.6f"
            % (prefix, peak, actual)
        )
    else:
        log.info(
            "threshold_list %s: Global peak = %.6f, sub-image clean threshold will be %.6f"
            % (prefix, peak, actual)
        )

    return actual
