""" Functions for tropospheric and ionospheric modeling
: see
`SDP Memo 97 <http://ska-sdp.org/sites/default/files/attachments/direction_dependent_self_calibration_in_arl_-_signed.pdf>`_


"""

__all__ = [
    "find_pierce_points",
    "create_gaintable_from_screen",
    "grid_gaintable_to_screen",
    "calculate_sf_from_screen",
    "plot_gaintable_on_screen",
]

import warnings
import astropy.units as units
import numpy
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import FITSFixedWarning
from astropy.wcs import WCS

from rascil.data_models.memory_data_models import BlockVisibility
from rascil.processing_components.calibration.operations import (
    create_gaintable_from_blockvisibility,
    create_gaintable_from_rows,
)
from rascil.processing_components.visibility.visibility_geometry import (
    calculate_blockvisibility_hourangles,
)
from rascil.processing_components.util.coordinate_support import (
    xyz_to_uvw,
    skycoord_to_lmn,
)

import logging

log = logging.getLogger("rascil-logger")


def find_pierce_points(station_locations, ha, dec, phasecentre, height):
    """Find the pierce points for a flat screen at specified height

    A pierce point is where the line of site from a station or dish to a source passes
    through a thin screen

    :param station_locations: station locations [:3]
    :param ha: Hour angle
    :param dec: Declination
    :param phasecentre: Phase centre
    :param height: Height of screen
    :return:
    """
    source_direction = SkyCoord(ra=ha, dec=dec, frame="icrs", equinox="J2000")
    local_locations = xyz_to_uvw(station_locations, ha, dec)
    local_locations -= numpy.average(local_locations, axis=0)

    lmn = numpy.array(skycoord_to_lmn(source_direction, phasecentre))
    lmn[2] += 1.0
    pierce_points = local_locations + height * numpy.array(lmn)
    return pierce_points


def create_gaintable_from_screen(
    vis,
    sc,
    screen,
    height=None,
    vis_slices=None,
    r0=5e3,
    type_atmosphere="ionosphere",
    reference_component=None,
    jones_type="B",
    **kwargs
):
    """Create gaintables from a screen calculated using ARatmospy

    Screen axes are ['XX', 'YY', 'TIME', 'FREQ']

    :param vis:
    :param sc: Sky components for which pierce points are needed
    :param screen: Image or string (for fits file which will be memory mapped in
    :param height: Height (in m) of screen above telescope e.g. 3e5
    :param r0: r0 in meters
    :param type_atmosphere: 'ionosphere' or 'troposphere'
    :param reference: Use the first component as a reference
    :param jones_type: Type of calibration matrix T or G or B
    :return:
    """
    # assert isinstance(vis, BlockVisibility)

    assert height is not None, "Screen height must be specified"

    station_locations = vis.configuration.xyz.data

    scale = numpy.power(r0 / 5000.0, -5.0 / 3.0)

    nant = station_locations.shape[0]
    t2r = numpy.pi / 43200.0
    gaintables = [
        create_gaintable_from_blockvisibility(vis, jones_type=jones_type, **kwargs)
        for i in sc
    ]

    # Use memmap to speed up access and limit memory use
    warnings.simplefilter("ignore", FITSFixedWarning)
    hdulist = fits.open(screen, memmap=True)
    screen_data = hdulist[0].data
    screen_wcs = WCS(screen)
    screen_freq = screen_wcs.wcs.crval[3]
    assert screen_data.shape[0] == 1, screen_data.shape

    number_bad = 0
    number_good = 0
    ncomp = len(sc)

    for icomp, comp in enumerate(sc):
        gt = gaintables[icomp]
        for row, time in enumerate(gt.time):
            time_slice = {
                "time": slice(time - gt.interval[row] / 2, time + gt.interval[row] / 2)
            }
            v = vis.sel(time_slice)
            ha = numpy.average(calculate_blockvisibility_hourangles(v).to("rad").value)
            scr = numpy.zeros([nant, vis.blockvisibility_acc.nchan])
            pp = find_pierce_points(
                station_locations,
                (comp.direction.ra.rad + ha) * units.rad,
                comp.direction.dec,
                height=height,
                phasecentre=vis.phasecentre,
            )
            for ant in range(nant):
                pp0 = pp[ant][0:2]
                try:
                    worldloc = [pp0[0], pp0[1], 43200.0 * ha / numpy.pi, 0]
                    pixloc = screen_wcs.wcs_world2pix([worldloc], 0).astype("int")[0]
                    if type_atmosphere == "ionosphere":
                        # In the ionosphere file, the units are dTEC.
                        dtec = screen_data[0, pixloc[2], pixloc[1], pixloc[0]]
                        scr[ant, :] = -(scale * 8.44797245e9 / v.frequency) * dtec
                    else:
                        # In troposphere files, the units are phase in radians at the reference frequency
                        phase = screen_data[0, pixloc[2], pixloc[1], pixloc[0]]
                        scr[ant, :] = -(v.frequency / screen_freq) * phase
                    number_good += 1
                except (ValueError, IndexError):
                    number_bad += 1
                    scr[ant, ...] = 0.0
            # axes of gaintable.gain are time, ant, nchan, nrec
            gt.gain.data[row, :, :, :] = numpy.exp(1j * scr[...])[
                ..., numpy.newaxis, numpy.newaxis
            ]
            if gt.gain.data.shape[-1] == 2:
                gt.gain.data[..., 0, 1] *= 0.0
                gt.gain.data[..., 1, 0] *= 0.0
            gt.attrs["phasecentre"] = comp.direction

        # if reference_component is not None:
        #     scr -= scr[reference_component, ...][numpy.newaxis, ...]

    assert (
        number_good > 0
    ), "create_gaintable_from_screen: There are no pierce points inside the atmospheric screen image"
    if number_bad > 0:
        log.info(
            "create_gaintable_from_screen: %d pierce points are inside the atmospheric screen image"
            % (number_good)
        )
        log.info(
            "create_gaintable_from_screen: %d pierce points are outside the atmospheric screen image"
            % (number_bad)
        )

    hdulist.close()

    return gaintables


def grid_gaintable_to_screen(
    vis,
    gaintables,
    screen,
    height=3e5,
    gaintable_slices=None,
    scale=1.0,
    r0=5e3,
    type_atmosphere="ionosphere",
    vis_slices=None,
    **kwargs
):
    """Grid a gaintable to a screen image

    Screen axes are ['XX', 'YY', 'TIME', 'FREQ']

    The phases are just averaged per grid cell, no phase unwrapping is performed.

    :param vis:
    :param gaintables: input gaintables
    :param screen:
    :param height: Height (in m) of screen above telescope e.g. 3e5
    :param r0: r0 in meters
    :param type_atmosphere: 'ionosphere' or 'troposphere'
    :param scale: Multiply the screen by this factor
    :return: gridded screen image, weights image
    """
    # assert isinstance(vis, BlockVisibility)

    station_locations = vis.configuration.xyz.data

    nant = station_locations.shape[0]
    t2r = numpy.pi / 43200.0

    newscreen = screen.copy(deep=True)
    newscreen["pixels"].data[...] = 0.0
    weights = screen.copy(deep=True)
    weights["pixels"].data[...] = 0.0
    nchan, ntimes, ny, nx = screen["pixels"].data.shape

    number_no_weight = 0

    for gt in gaintables:
        ha_zero = numpy.average(calculate_blockvisibility_hourangles(vis))
        for row, time in enumerate(gt.time):
            time_slice = {
                "time": slice(time - gt.interval[row] / 2, time + gt.interval[row] / 2)
            }
            v = vis.sel(time_slice)
            ha = (
                numpy.average(calculate_blockvisibility_hourangles(v) - ha_zero)
                .to("rad")
                .value
            )
            pp = find_pierce_points(
                station_locations,
                (gt.phasecentre.ra.rad + ha) * units.rad,
                gt.phasecentre.dec,
                height=height,
                phasecentre=vis.phasecentre,
            )

            scr = numpy.angle(gt.gain[0, :, 0, 0, 0])
            wt = gt.weight[0, :, 0, 0, 0]
            for ant in range(nant):
                pp0 = pp[ant][0:2]
                for freq in vis.frequency:
                    scale = numpy.power(r0 / 5000.0, -5.0 / 3.0)
                    if type_atmosphere == "troposphere":
                        # In troposphere files, the units are phase in radians.
                        screen_to_phase = scale
                    else:
                        # In the ionosphere file, the units are dTEC.
                        screen_to_phase = -scale * 8.44797245e9 / freq
                    worldloc = [pp0[0], pp0[1], 43200 * ha / numpy.pi, freq]
                    pixloc = newscreen.wcs.wcs_world2pix([worldloc], 0)[0].astype("int")
                    assert pixloc[0] >= 0
                    assert pixloc[0] < nx
                    assert pixloc[1] >= 0
                    assert pixloc[1] < ny
                    pixloc[3] = 0
                    newscreen["pixels"].data[
                        pixloc[3], pixloc[2], pixloc[1], pixloc[0]
                    ] += (wt[ant] * scr[ant] / screen_to_phase)
                    weights["pixels"].data[
                        pixloc[3], pixloc[2], pixloc[1], pixloc[0]
                    ] += wt[ant]
                    if wt[ant] == 0.0:
                        number_no_weight += 1
    if number_no_weight > 0:
        log.warning(
            "grid_gaintable_to_screen: %d pierce points are have no weight"
            % (number_no_weight)
        )

    assert numpy.max(weights["pixels"].data) > 0.0, "No points were gridded"

    newscreen["pixels"].data[weights["pixels"].data > 0.0] = (
        newscreen["pixels"].data[weights["pixels"].data > 0.0]
        / weights["pixels"].data[weights["pixels"].data > 0.0]
    )

    return newscreen, weights


def calculate_sf_from_screen(screen):
    """Calculate structure function image from screen

    Screen axes are ['XX', 'YY', 'TIME', 'FREQ']

    :param screen:
    :return:
    """
    from scipy.signal import fftconvolve

    nchan, ntimes, ny, nx = screen["pixels"].data.shape

    sf = numpy.zeros([nchan, 1, 2 * ny - 1, 2 * nx - 1])
    for chan in range(nchan):
        sf[chan, 0, ...] = fftconvolve(
            screen.data[chan, 0, ...], screen.data[chan, 0, ::-1, ::-1]
        )
        for itime in range(ntimes):
            sf += fftconvolve(
                screen.data[chan, itime, ...], screen.data[chan, itime, ::-1, ::-1]
            )
        sf[chan, 0, ...] /= numpy.max(sf[chan, 0, ...])
        sf[chan, 0, ...] = 1.0 - sf[chan, 0, ...]

    sf_image = screen.copy(deep=True)
    sf_image["pixels"].data = sf[
        :, :, (ny - ny // 4) : (ny + ny // 4), (nx - nx // 4) : (nx + nx // 4)
    ]
    sf_image.wcs.wcs.crpix[0] = ny // 4 + 1
    sf_image.wcs.wcs.crpix[1] = ny // 4 + 1
    sf_image.wcs.wcs.crpix[2] = 1

    return sf_image


def plot_gaintable_on_screen(
    vis, gaintables, height=3e5, gaintable_slices=None, plotfile=None
):
    """Plot a gaintable on an ionospheric screen

    Screen axes are ['XX', 'YY', 'TIME', 'FREQ']

    :param vis:
    :param gaintables:
    :param height: Height (in m) of screen above telescope e.g. 3e5
    :param scale: Multiply the screen by this factor
    :return: gridded screen image, weights image
    """

    import matplotlib.pyplot as plt

    # assert isinstance(vis, BlockVisibility)

    station_locations = vis.configuration.xyz.data

    t2r = numpy.pi / 43200.0

    # The time in the BlockVisibility is UTC in seconds
    plt.clf()
    for gt in gaintables:
        time_zero = numpy.average(gt.time)
        for row, time in enumerate(gt.time):
            time_slice = {
                "time": slice(time - gt.interval[row] / 2, time + gt.interval[row] / 2)
            }
            gt_sel = gt.sel(time_slice)
            ha = numpy.average(gt_sel.time - time_zero)

            pp = find_pierce_points(
                station_locations,
                (gt_sel.phasecentre.ra.rad + t2r * ha) * units.rad,
                gt_sel.phasecentre.dec,
                height=height,
                phasecentre=vis.phasecentre,
            )
            phases = numpy.angle(gt_sel.gain[0, :, 0, 0, 0])
            plt.scatter(pp[:, 0], pp[:, 1], c=phases, cmap="hsv", alpha=0.75, s=0.1)

    plt.title("Pierce point phases")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    if plotfile is not None:
        plt.savefig(plotfile)

    plt.show(block=False)
