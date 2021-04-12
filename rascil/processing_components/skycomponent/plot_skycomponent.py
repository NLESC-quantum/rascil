"""Functions to manage plotting skycomponents in comparisons.

"""

__all__ = [
    "plot_skycomponents_positions",
    "plot_skycomponents_position_distance",
    "plot_skycomponents_flux",
    "plot_skycomponents_flux_ratio",
    "plot_skycomponents_flux_histogram",
    "plot_skycomponents_position_quiver",
    "plot_gaussian_beam_position",
]

import collections
import logging

import matplotlib.pyplot as plt

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord
from rascil.data_models.memory_data_models import Skycomponent
from rascil.processing_components.skycomponent.operations import (
    find_skycomponent_matches,
    fit_skycomponent,
)

log = logging.getLogger("rascil-logger")


def plot_skycomponents_positions(
    comps_test,
    comps_ref=None,
    img_size=1.0,
    plot_file=None,
    tol=1e-5,
    plot_error=True,
    **kwargs,
):
    """Generate position scatter plot for two lists of skycomponents

    :param comps_test: List of components to be tested
    :param img_size: Cell size per pixel in the image to compare
    :param comps_ref: List of reference components
    :param plot_file: Filename of the plot
    :param tol: Tolerance in rad
    :param plot_error : If True, plot error, else just plot absolute values
    :return: [ra_error, dec_error]:
             The error array for users to check
    """
    angle_wrap = 180.0 * u.deg

    if comps_ref is None:  # No comparison needed
        ra_test = [comp.direction.ra.wrap_at(angle_wrap).degree for comp in comps_test]
        dec_test = [comp.direction.dec.degree for comp in comps_test]

        plt.plot(ra_test, dec_test, "o", color="b", markersize=5, label="Components")

    else:

        matches = find_skycomponent_matches(comps_test, comps_ref, tol)
        ra_test = numpy.zeros(len(matches))
        dec_test = numpy.zeros(len(matches))
        ra_ref = numpy.zeros(len(matches))
        dec_ref = numpy.zeros(len(matches))
        ra_error = numpy.zeros(len(matches))
        dec_error = numpy.zeros(len(matches))
        for i, match in enumerate(matches):
            m_comp = comps_test[match[0]]
            ra_test[i] = m_comp.direction.ra.wrap_at(angle_wrap).degree
            dec_test[i] = m_comp.direction.dec.degree
            m_ref = comps_ref[match[1]]
            ra_ref[i] = m_ref.direction.ra.wrap_at(angle_wrap).degree
            dec_ref[i] = m_ref.direction.dec.degree

            if img_size > 0.0:
                ra_error[i] = (
                    (
                        m_comp.direction.ra.wrap_at(angle_wrap).degree
                        - m_ref.direction.ra.wrap_at(angle_wrap).degree
                    )
                    * numpy.cos(m_ref.direction.dec.rad)
                    / img_size
                )

                dec_error[i] = (
                    m_comp.direction.dec.degree - m_ref.direction.dec.degree
                ) / img_size

            else:
                log.info("Wrong input image resolution. Plot absolute values instead.")
                ra_error[i] = (
                    m_comp.direction.ra.wrap_at(angle_wrap).degree
                    - m_ref.direction.ra.wrap_at(angle_wrap).degree
                ) * numpy.cos(m_ref.direction.dec.rad)

                dec_error[i] = m_comp.direction.dec.degree - m_ref.direction.dec.degree

        ax = plt.gca()
        ax.set_aspect(1.0)
        ax.plot(
            ra_test, dec_test, "o", color="b", markersize=5, label="Tested components"
        )
        ax = plt.gca()
        ax.set_aspect(1.0)
        ax.plot(
            ra_ref, dec_ref, "x", color="r", markersize=8, label="Original components"
        )

    plt.title("Positions of sources")
    plt.xlabel("RA (deg)")
    plt.ylabel("Dec (deg)")
    plt.legend(loc="best")
    if plot_file is not None:
        plt.savefig(plot_file + "_position_value.png")
    plt.show(block=False)
    plt.clf()

    if plot_error is True:
        if comps_ref is None:
            log.info("Error: No reference components. No position errors are plotted.")
        else:
            ax = plt.gca()
            ax.set_aspect(1.0)
            ax.plot(ra_error, dec_error, "o", markersize=5)
        err_r = max(numpy.max(ra_error), numpy.max(dec_error))
        err_l = min(numpy.min(ra_error), numpy.min(dec_error))
        plt.xlim([err_l, err_r])
        plt.ylim([err_l, err_r])
        plt.xlabel(r"$\Delta\ RA * cos(Dec) / \Delta x$")
        plt.ylabel(r"$\Delta\ Dec/ \Delta x$")
        plt.title("Errors in RA and Dec")
        if plot_file is not None:
            plt.savefig(plot_file + "_position_error.png")
        plt.show(block=False)
        plt.clf()

    return [ra_test, dec_test]


def plot_skycomponents_position_distance(
    comps_test, comps_ref, phasecentre, img_size, plot_file=None, tol=1e-5, **kwargs
):
    """Generate position error plot vs distance for two lists of skycomponents

    :param comps_test: List of components to be tested
    :param comps_ref: List of reference components
    :param plot_file: Filename of the plot
    :param tol: Tolerance in rad
    :param phasecentre: Centre of image in SkyCoords
    :param img_size: Cell size per pixel in the image to compare
    :return: [ra_error, dec_error]:
             The error array for users to check
    """

    matches = find_skycomponent_matches(comps_test, comps_ref, tol)
    ra_error = numpy.zeros(len(matches))
    dec_error = numpy.zeros(len(matches))
    dist = numpy.zeros(len(matches))
    for i, match in enumerate(matches):
        m_comp = comps_test[match[0]]
        m_ref = comps_ref[match[1]]

        if img_size > 0.0:
            ra_error[i] = (
                (m_comp.direction.ra.degree - m_ref.direction.ra.degree)
                * numpy.cos(m_ref.direction.dec.rad)
                / img_size
            )
            dec_error[i] = (
                m_comp.direction.dec.degree - m_ref.direction.dec.degree
            ) / img_size

            dist[i] = m_comp.direction.separation(phasecentre).degree
        else:
            log.info("Wrong input image resolution. Plot absolute values instead.")
            ra_error[i] = (
                m_comp.direction.ra.degree - m_ref.direction.ra.degree
            ) * numpy.cos(m_ref.direction.dec.rad)

            dec_error[i] = m_comp.direction.dec.degree - m_ref.direction.dec.degree
            dist[i] = m_comp.direction.separation(phasecentre).degree

    err_r = max(numpy.max(ra_error), numpy.max(dec_error))
    err_l = min(numpy.min(ra_error), numpy.min(dec_error))

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle("Position error vs. Distance")
    ax1.plot(dist, ra_error, "o", color="b", markersize=5)
    ax2.plot(dist, dec_error, "o", color="b", markersize=5)

    ax1.set_ylabel(r"$\Delta\ RA/ \Delta x$")
    ax2.set_ylabel(r"$\Delta\ Dec/ \Delta x$")
    ax2.set_xlabel(r"Separation To Center (deg)")
    ax1.set_ylim([err_l, err_r])
    ax2.set_ylim([err_l, err_r])
    if plot_file is not None:
        plt.savefig(plot_file + "_position_distance.png")
    plt.show(block=False)
    plt.clf()

    return [ra_error, dec_error]


def plot_skycomponents_flux(
    comps_test, comps_ref, plot_file=None, tol=1e-5, refchan=None, **kwargs
):
    """Generate flux scatter plot for two lists of skycomponents

    :param comps_test: List of components to be tested
    :param comps_ref: List of reference components
    :param plot_file: Filename of the plot
    :param tol: Tolerance in rad
    :param refchan: Reference channel for comparison, default is centre channel
    :return: [flux_in, flux_out]:
             The flux array for users to check
    """

    matches = find_skycomponent_matches(comps_test, comps_ref, tol)
    flux_in = numpy.zeros(len(matches))
    flux_out = numpy.zeros(len(matches))
    for i, match in enumerate(matches):
        m_comp = comps_test[match[0]]
        m_ref = comps_ref[match[1]]
        if refchan is None:
            nchan, _ = m_ref.flux.shape
            flux_in[i] = m_ref.flux[nchan // 2]
            flux_out[i] = m_comp.flux[0]
        else:
            flux_in[i] = m_ref.flux[refchan]
            flux_out[i] = m_comp.flux[0]

    plt.loglog(flux_in, flux_out, "o", color="b", markersize=5)

    plt.title("Flux in vs. flux out")
    plt.xlabel("Flux in (Jy)")
    plt.ylabel("Flux out (Jy)")
    if plot_file is not None:
        plt.savefig(plot_file + "_flux_value.png")
    plt.show(block=False)
    plt.clf()

    return [flux_in, flux_out]


def plot_skycomponents_flux_ratio(
    comps_test,
    comps_ref,
    phasecentre,
    plot_file=None,
    tol=1e-5,
    refchan=None,
    max_ratio=2,
    **kwargs,
):

    """Generate flux ratio plot vs distance for two lists of skycomponents

    :param comps_test: List of components to be tested
    :param comps_ref: List of reference components
    :param plot_file: Filename of the plot
    :param tol: Tolerance in rad
    :param phasecentre: Centre of image in SkyCoords
    :param refchan: Reference channel for comparison, default is centre channel
    :param max_ratio: Maximum ratio to plot (default is 2.0)
    :return: [dist, flux_ratio]:
             The flux array for users to check
    """

    matches = find_skycomponent_matches(comps_test, comps_ref, tol)
    flux_ratio = []
    dist = []
    for i, match in enumerate(matches):
        m_comp = comps_test[match[0]]
        m_ref = comps_ref[match[1]]

        if m_ref.flux[0] > 0.0:
            if refchan is None:
                nchan, _ = m_ref.flux.shape
                fr = m_comp.flux[0] / m_ref.flux[nchan // 2]
            else:
                fr = m_comp.flux[0] / m_ref.flux[refchan]
            if fr < max_ratio:
                flux_ratio.append(fr)
                dist.append(m_comp.direction.separation(phasecentre).degree)

    if len(dist) == 0:
        raise ValueError("No valid points found for flux ratio plot")

    plt.plot(dist, flux_ratio, "o", color="b", markersize=5)

    plt.title("Flux ratio vs. distance")
    plt.xlabel("Distance to center (Deg)")
    plt.ylabel("Flux Ratio (Out/In)")
    if plot_file is not None:
        plt.savefig(plot_file + "_flux_ratio.png")
    plt.show(block=False)
    plt.clf()

    return [dist, flux_ratio]


def plot_skycomponents_flux_histogram(
    comps_test, comps_ref, plot_file=None, nbins=10, tol=1e-5, refchan=None, **kwargs
):

    """Generate flux ratio plot vs distance for two lists of skycomponents

    :param comps_test: List of components to be tested
    :param comps_ref: List of reference components
    :param plot_file: Filename of the plot
    :param tol: Tolerance in rad
    :param nbins: Number of bins for the histrogram
    :param refchan: Reference channel for comparison, default is centre channel
    :return: hist: The flux array for users to check
    """

    if refchan is None:
        nchan, _ = comps_ref[0].flux.shape
        flux_in = numpy.array([comp.flux[nchan // 2, 0] for comp in comps_ref])
        flux_out = numpy.array([comp.flux[0, 0] for comp in comps_test])
    else:
        flux_in = numpy.array([comp.flux[refchan, 0] for comp in comps_ref])
        flux_out = numpy.array([comp.flux[0, 0] for comp in comps_test])

    flux_in = flux_in[flux_in > 0.0]
    flux_out = flux_out[flux_out > 0.0]

    hist = [flux_in, flux_out]
    labels = ["Flux In", "Flux Out"]
    colors = ["r", "b"]
    hist_min = min(numpy.min(flux_in), numpy.min(flux_out))
    hist_max = max(numpy.max(flux_in), numpy.max(flux_out))

    hist_bins = numpy.logspace(numpy.log10(hist_min), numpy.log10(hist_max), nbins)

    fig, ax = plt.subplots()
    ax.hist(hist, bins=hist_bins, log=True, color=colors, label=labels)

    ax.set_title("Flux histogram")
    ax.set_xlabel("Flux (Jy)")
    ax.set_xscale("log")
    ax.set_ylabel("Source Count")
    plt.legend(loc="best")
    if plot_file is not None:
        plt.savefig(plot_file + "_flux_histogram.png")
    plt.show(block=False)
    plt.clf()

    return hist


def plot_skycomponents_position_quiver(
    comps_test, comps_ref, phasecentre, num=100, plot_file=None, tol=1e-5, **kwargs
):
    """Generate position error quiver diagram for two lists of skycomponents

    :param comps_test: List of components to be tested
    :param comps_ref: List of reference components
    :param phasecentre: Centre of image in SkyCoords
    :param num: Number of the brightest sources to plot
    :param plot_file: Filename of the plot
    :param tol: Tolerance in rad
    :return: [ra_error, dec_error]:
             The error array for users to check
    """

    angle_wrap = 180.0 * u.deg

    comps_test_sorted = sorted(comps_test, key=lambda cmp: numpy.max(cmp.flux))

    matches = find_skycomponent_matches(comps_test_sorted, comps_ref, tol)
    num = min(num, len(matches))
    ra_ref = numpy.zeros(num)
    dec_ref = numpy.zeros(num)
    ra_error = numpy.zeros(num)
    dec_error = numpy.zeros(num)

    for i in range(num):
        m_comp = comps_test_sorted[matches[i][0]]
        m_ref = comps_ref[matches[i][1]]

        ra_ref[i] = m_ref.direction.ra.wrap_at(angle_wrap).degree
        dec_ref[i] = m_ref.direction.dec.degree
        ra_error[i] = (
            m_comp.direction.ra.wrap_at(angle_wrap).degree
            - m_ref.direction.ra.wrap_at(angle_wrap).degree
        ) * numpy.cos(m_ref.direction.dec.rad)

        dec_error[i] = m_comp.direction.dec.degree - m_ref.direction.dec.degree

    ref = max(numpy.max(numpy.abs(ra_error)), numpy.max(numpy.abs(dec_error)))
    scale_factor = 10 * ref
    log.info(f" Scale factor is {scale_factor}")
    fig, ax = plt.subplots()
    if numpy.mean(numpy.deg2rad(dec_ref)) != 0.0:
        ax.set_aspect(1.0 / numpy.cos(numpy.mean(numpy.deg2rad(dec_ref))))
    q = ax.quiver(ra_ref, dec_ref, ra_error, dec_error, color="b")

    ax.scatter(ra_ref, dec_ref, color="r", s=8)
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    plt.title(f"Brightest {num} sources")
    if plot_file is not None:
        plt.savefig(plot_file + "_position_quiver.png")
    plt.show(block=False)
    plt.clf()

    return [ra_error, dec_error]


def plot_gaussian_beam_position(
    comps_test,
    comps_ref,
    phasecentre,
    image,
    num=100,
    plot_file=None,
    tol=1e-5,
    **kwargs,
):
    """Plot the major and minor size of beams for two lists of skycomponents
    :param comps_test: List of components to be tested
    :param comps_ref: List of reference components
    :param phasecentre: Centre of image in SkyCoords
    :param image: Image to fit the skycomponents
    :param num: Number of the brightest sources to plot
    :param plot_file: Filename of the plot
    :param tol: Tolerance in rad

    :return: [bmaj, bmin]:
             The beam parameters for users to check
    """

    angle_wrap = 180.0 * u.deg

    comps_test_sorted = sorted(comps_test, key=lambda cmp: numpy.max(cmp.flux))
    matches = find_skycomponent_matches(comps_test_sorted, comps_ref, tol)
    num = min(num, len(matches))

    # Only put in the items that can be fitted
    ra_dist = numpy.zeros(num)
    dec_dist = numpy.zeros(num)
    bmaj = numpy.zeros(num)
    bmin = numpy.zeros(num)
    dist = numpy.zeros(num)

    count = 0
    i = 0
    while count < num:
        match = matches[i]
        m_comp = comps_test_sorted[match[0]]
        m_ref = comps_ref[match[1]]

        log.info(f"Processing {match[0]}")
        i = i + 1
        try:
            fitted = fit_skycomponent(image, m_comp, force_point_sources=False)
            log.info("{}".format(fitted.params))
            ra_dist[count] = (m_comp.direction.ra.wrap_at(angle_wrap).degree -
                              phasecentre.ra.wrap_at(angle_wrap).deg)\
                             * numpy.cos(m_ref.direction.dec.rad)
            dec_dist[count] = (m_comp.direction.dec.degree - phasecentre.dec.deg)
            dist[count] = m_comp.direction.separation(phasecentre).degree
            bmaj[count] = fitted.params["bmaj"]
            bmin[count] = fitted.params["bmin"]
            count = count + 1

	#If fitting failed, no items will be found in the params dictionary
        except KeyError as err:
            log.warning(f"Fit skycomponent failed for component number {match[0]} ")

    log.info(f"Fitted {i} components, selected {num}")

    pos_dist = numpy.sqrt(numpy.array(ra_dist) ** 2.0 + numpy.array(dec_dist) ** 2.0)

    dist_r = numpy.max(pos_dist)
    dist_l = numpy.min(pos_dist)

    beam_r = max(numpy.max(bmaj), numpy.max(bmin))
    beam_l = min(numpy.min(bmin), numpy.min(bmin))

    ax1 = plt.subplot(212)
    ax1.plot(dist, bmaj, "o", color="b", markersize=5, label="Bmaj")
    ax1.plot(dist, bmin, "o", color="r", markersize=5, label="Bmin")
    ax1.legend(loc="best")
    ax1.set_ylabel("Beam size (deg)")
    ax1.set_xlabel(r"Distance to phase centre (deg)")

    ax2 = plt.subplot(221)
    ax2.plot(ra_dist, bmaj, "o", color="b", markersize=5, label="Bmaj")
    ax2.plot(ra_dist, bmin, "o", color="r", markersize=5, label="Bmin")
    ax2.set_ylabel(r"Beam size (deg)")
    ax2.set_title(r"$\Delta RA (deg)$")
    ax2.set_xlim([dist_l, dist_r])
    ax2.set_ylim([beam_l, beam_r])

    ax3 = plt.subplot(222)
    ax3.plot(dec_dist, bmaj, "o", color="b", markersize=5, label="Bmaj")
    ax3.plot(dec_dist, bmin, "o", color="r", markersize=5, label="Bmin")
    ax3.set_title(r"$\Delta Dec (deg)$")
    ax3.set_xlim([dist_l, dist_r])
    ax3.set_ylim([beam_l, beam_r])

    if plot_file is not None:
        plt.savefig(plot_file + "_gaussian_beam_position.png")
    plt.show(block=False)
    plt.clf()

    return [bmaj, bmin]
