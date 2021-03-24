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
import matplotlib

matplotlib.use("Agg")
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

    :param comps_test: List of components to be test
    :param comps_ref: List of reference components
    :param plot_file: Filename of the plot
    :param tol: Tolerance in rad
    :param plot_error : Whether to plot absolute values or error
    :return:
    """
    if comps_ref is None:  # No comparison needed
        ra_test = [comp.direction.ra.radian for comp in comps_test]
        dec_test = [comp.direction.dec.radian for comp in comps_test]

        plt.plot(ra_test, dec_test, ".", color="b", markersize=5, label="Components")

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
            ra_test[i] = m_comp.direction.ra.radian
            dec_test[i] = m_comp.direction.dec.radian
            m_ref = comps_ref[match[1]]
            ra_ref[i] = m_ref.direction.ra.radian
            dec_ref[i] = m_ref.direction.dec.radian
            ra_error[i] = numpy.abs(
                m_comp.direction.ra.radian - m_ref.direction.ra.radian
            )
            dec_error[i] = numpy.abs(
                m_comp.direction.dec.radian - m_ref.direction.dec.radian
            )

        plt.plot(
            ra_test, dec_test, ".", color="b", markersize=5, label="Tested components"
        )
        plt.plot(
            ra_ref, dec_ref, ".", color="r", markersize=5, label="Original components"
        )

    plt.xlabel("RA (rad)")
    plt.ylabel("Dec (rad)")
    plt.legend(loc="best")
    if plot_file is not None:
        plt.savefig(plot_file + "_position_value.png")
    plt.show(block=False)
    plt.clf()

    if plot_error is True:
        if comps_ref is None:
            log.info("Error: No reference components. No position errors are plotted.")
        else:
            plt.plot(ra_error, dec_error, ".", markersize=5)
        plt.xlabel("RA (rad)")
        plt.ylabel("Dec (rad)")
        plt.title("Errors in RA and Dec")
        if plot_file is not None:
            plt.savefig(plot_file + "_position_error.png")
        plt.show(block=False)
        plt.clf()


def plot_skycomponents_position_distance(
    comps_test, comps_ref, phasecentre, plot_file=None, tol=1e-5, **kwargs
):
    """Generate position error plot vs distance for two lists of skycomponents

    :param comps_test: List of components to be test
    :param comps_ref: List of reference components
    :param plot_file: Filename of the plot
    :param tol: Tolerance in rad
    :param phasecentre: Centre of image in SkyCoords
    :return:
    """

    matches = find_skycomponent_matches(comps_test, comps_ref, tol)
    ra_error = numpy.zeros(len(matches))
    dec_error = numpy.zeros(len(matches))
    dist = numpy.zeros(len(matches))
    for i, match in enumerate(matches):
        m_comp = comps_test[match[0]]
        m_ref = comps_ref[match[1]]

        ra_error[i] = numpy.abs(m_comp.direction.ra.radian - m_ref.direction.ra.radian)
        dec_error[i] = numpy.abs(
            m_comp.direction.dec.radian - m_ref.direction.dec.radian
        )

        dist[i] = m_comp.direction.separation(phasecentre).rad

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle("Position error vs. Distance")
    ax1.plot(dist, ra_error, ".", color="b", markersize=5)
    ax2.plot(dist, dec_error, ".", color="b", markersize=5)

    ax1.set_ylabel("RA (rad)")
    ax2.set_ylabel("Dec (rad)")
    ax2.set_xlabel("Separation To Center(rad)")
    if plot_file is not None:
        plt.savefig(plot_file + "_position_distance.png")
    plt.show(block=False)
    plt.clf()


def plot_skycomponents_flux(comps_test, comps_ref, plot_file=None, tol=1e-5, **kwargs):
    """Generate flux scatter plot for two lists of skycomponents

    :param comps_test: List of components to be test
    :param comps_ref: List of reference components
    :param plot_file: Filename of the plot
    :param tol: Tolerance in rad
    :return:
    """

    matches = find_skycomponent_matches(comps_test, comps_ref, tol)
    flux_in = numpy.zeros(len(matches))
    flux_out = numpy.zeros(len(matches))
    for i, match in enumerate(matches):
        m_comp = comps_test[match[0]]
        m_ref = comps_ref[match[1]]
        flux_in[i] = m_ref.flux[0]
        flux_out[i] = m_comp.flux[0]

    plt.plot(flux_in, flux_out, ".", color="b", markersize=5)

    plt.xlabel("Flux in (Jy)")
    plt.ylabel("Flux out (Jy)")
    if plot_file is not None:
        plt.savefig(plot_file + "_flux_value.png")
    plt.show(block=False)
    plt.clf()


def plot_skycomponents_flux_ratio(
    comps_test, comps_ref, phasecentre, plot_file=None, tol=1e-5, **kwargs
):

    """Generate flux ratio plot vs distance for two lists of skycomponents

    :param comps_test: List of components to be test
    :param comps_ref: List of reference components
    :param plot_file: Filename of the plot
    :param tol: Tolerance in rad
    :param phasecentre: Centre of image in SkyCoords
    :return:
    """

    matches = find_skycomponent_matches(comps_test, comps_ref, tol)
    flux_ratio = numpy.zeros(len(matches))
    dist = numpy.zeros(len(matches))
    for i, match in enumerate(matches):
        m_comp = comps_test[match[0]]
        m_ref = comps_ref[match[1]]
        if m_ref.flux[0] == 0.:
            flux_ratio[i] = 0.
        else:
            flux_ratio[i] = m_comp.flux[0] / m_ref.flux[0]
        dist[i] = m_comp.direction.separation(phasecentre).rad

    plt.plot(dist, flux_ratio, ".", color="b", markersize=5)

    plt.xlabel("Separation to center (Rad)")
    plt.ylabel("Flux Ratio")
    if plot_file is not None:
        plt.savefig(plot_file + "_flux_ratio.png")
    plt.show(block=False)
    plt.clf()


def plot_skycomponents_flux_histogram(
    comps_test, comps_ref, plot_file=None, nbins=10, tol=1e-5, **kwargs
):

    """Generate flux ratio plot vs distance for two lists of skycomponents

    :param comps_test: List of components to be test
    :param comps_ref: List of reference components
    :param plot_file: Filename of the plot
    :param tol: Tolerance in rad
    :param nbins: number of bins for the histrogram
    :return:
    """

    flux_in = [comp.flux[0,0] for comp in comps_test]
    flux_out = [comp.flux[0,0] for comp in comps_ref]

    fig, ax = plt.subplots()
    ax.hist(flux_in, bins=nbins, log=True, color="b", label="Flux In")
    ax.hist(flux_out, bins=nbins, log=True, color="r", label="Flux Out")

    ax.set_xlabel("Flux (Jy)")
    ax.set_ylabel("Source Count")
    plt.legend(loc="best")
    if plot_file is not None:
        plt.savefig(plot_file + "_flux_histogram.png")
    plt.show(block=False)
    plt.clf()
