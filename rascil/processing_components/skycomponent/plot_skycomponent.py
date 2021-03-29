"""Functions to manage plotting skycomponents in comparisons.

"""

__all__ = [
    "plot_skycomponents_positions",
    "plot_skycomponents_position_distance",
    "plot_skycomponents_flux",
    "plot_skycomponents_flux_ratio",
    "plot_skycomponents_flux_histogram",
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
)

log = logging.getLogger("rascil-logger")


def plot_skycomponents_positions(
    comps_test, comps_ref=None, plot_file=None, tol=1e-5, plot_error=True, **kwargs
):
    """Generate position scatter plot for two lists of skycomponents

    :param comps_test: List of components to be tested
    :param comps_ref: List of reference components
    :param plot_file: Filename of the plot
    :param tol: Tolerance in rad
    :param plot_error : If True, plot error, else just plot absolute values
    :return: [ra_error, dec_error]:
             The error array for users to check
    """
    if comps_ref is None:  # No comparison needed
        ra_test = [comp.direction.ra.degree for comp in comps_test]
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
            ra_test[i] = m_comp.direction.ra.degree
            dec_test[i] = m_comp.direction.dec.degree
            m_ref = comps_ref[match[1]]
            ra_ref[i] = m_ref.direction.ra.degree
            dec_ref[i] = m_ref.direction.dec.degree
            ra_error[i] = m_comp.direction.ra.degree - m_ref.direction.ra.degree

            dec_error[i] = m_comp.direction.dec.degree - m_ref.direction.dec.degree

        plt.plot(
            ra_test, dec_test, "o", color="b", markersize=5, label="Tested components"
        )
        plt.plot(
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
            plt.plot(ra_error, dec_error, "o", markersize=5)
        err_r = max(numpy.max(ra_error), numpy.max(dec_error))
        err_l = min(numpy.min(ra_error), numpy.min(dec_error))
        plt.xlim([err_l, err_r])
        plt.ylim([err_l, err_r])
        plt.xlabel(r"$\Delta\ RA\ (deg)$")
        plt.ylabel(r"$\Delta\ Dec\ (deg)$")
        plt.title("Errors in RA and Dec")
        if plot_file is not None:
            plt.savefig(plot_file + "_position_error.png")
        plt.show(block=False)
        plt.clf()

    return [ra_test, dec_test]


def plot_skycomponents_position_distance(
    comps_test, comps_ref, phasecentre, plot_file=None, tol=1e-5, **kwargs
):
    """Generate position error plot vs distance for two lists of skycomponents

    :param comps_test: List of components to be tested
    :param comps_ref: List of reference components
    :param plot_file: Filename of the plot
    :param tol: Tolerance in rad
    :param phasecentre: Centre of image in SkyCoords
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

        ra_error[i] = m_comp.direction.ra.degree - m_ref.direction.ra.degree
        dec_error[i] = m_comp.direction.dec.degree - m_ref.direction.dec.degree

        dist[i] = m_comp.direction.separation(phasecentre).degree

    err_r = max(numpy.max(ra_error), numpy.max(dec_error))
    err_l = min(numpy.min(ra_error), numpy.min(dec_error))

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle("Position error vs. Distance")
    ax1.plot(dist, ra_error, "o", color="b", markersize=5)
    ax2.plot(dist, dec_error, "o", color="b", markersize=5)

    ax1.set_ylabel("RA (deg)")
    ax2.set_ylabel("Dec (deg)")
    ax2.set_xlabel("Separation To Center(deg)")
    ax1.set_ylim([err_l, err_r])
    ax2.set_ylim([err_l, err_r])
    if plot_file is not None:
        plt.savefig(plot_file + "_position_distance.png")
    plt.show(block=False)
    plt.clf()

    return [ra_error, dec_error]


def plot_skycomponents_flux(comps_test, comps_ref, plot_file=None, tol=1e-5, **kwargs):
    """Generate flux scatter plot for two lists of skycomponents

    :param comps_test: List of components to be tested
    :param comps_ref: List of reference components
    :param plot_file: Filename of the plot
    :param tol: Tolerance in rad
    :return: [flux_in, flux_out]:
             The flux array for users to check
    """

    matches = find_skycomponent_matches(comps_test, comps_ref, tol)
    flux_in = numpy.zeros(len(matches))
    flux_out = numpy.zeros(len(matches))
    for i, match in enumerate(matches):
        m_comp = comps_test[match[0]]
        m_ref = comps_ref[match[1]]
        flux_in[i] = m_ref.flux[0]
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
    comps_test, comps_ref, phasecentre, plot_file=None, tol=1e-5, **kwargs
):

    """Generate flux ratio plot vs distance for two lists of skycomponents

    :param comps_test: List of components to be tested
    :param comps_ref: List of reference components
    :param plot_file: Filename of the plot
    :param tol: Tolerance in rad
    :param phasecentre: Centre of image in SkyCoords
    :return: [dist, flux_ratio]:
             The flux array for users to check
    """

    matches = find_skycomponent_matches(comps_test, comps_ref, tol)
    flux_ratio = numpy.zeros(len(matches))
    dist = numpy.zeros(len(matches))
    for i, match in enumerate(matches):
        m_comp = comps_test[match[0]]
        m_ref = comps_ref[match[1]]
        if m_ref.flux[0] == 0.0:
            flux_ratio[i] = 0.0
        else:
            flux_ratio[i] = m_comp.flux[0] / m_ref.flux[0]
        dist[i] = m_comp.direction.separation(phasecentre).degree

    plt.plot(dist, flux_ratio, "o", color="b", markersize=5)

    plt.title("Flux ratio vs. distance")
    plt.xlabel("Separation to center (Deg)")
    plt.ylabel("Flux Ratio")
    if plot_file is not None:
        plt.savefig(plot_file + "_flux_ratio.png")
    plt.show(block=False)
    plt.clf()

    return [dist, flux_ratio]


def plot_skycomponents_flux_histogram(
    comps_test, comps_ref, plot_file=None, nbins=10, tol=1e-5, **kwargs
):

    """Generate flux ratio plot vs distance for two lists of skycomponents

    :param comps_test: List of components to be tested
    :param comps_ref: List of reference components
    :param plot_file: Filename of the plot
    :param tol: Tolerance in rad
    :param nbins: Number of bins for the histrogram
    :return: hist: The flux array for users to check
    """

    flux_in = numpy.array([comp.flux[0, 0] for comp in comps_ref])
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
