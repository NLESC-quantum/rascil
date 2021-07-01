"""Function to manage sky components.

"""

__all__ = [
    "create_skycomponent",
    "filter_skycomponents_by_flux",
    "filter_skycomponents_by_index",
    "find_nearest_skycomponent",
    "find_nearest_skycomponent_index",
    "find_separation_skycomponents",
    "find_skycomponent_matches",
    "find_skycomponent_matches_atomic",
    "find_skycomponents",
    "fit_skycomponent",
    "fit_skycomponent_spectral_index",
    "insert_skycomponent",
    "voronoi_decomposition",
    "image_voronoi_iter",
    "partition_skycomponent_neighbours",
    "select_components_by_separation",
    "select_neighbouring_components",
    "remove_neighbouring_components",
    "apply_beam_to_skycomponent",
    "apply_voltage_pattern_to_skycomponent",
]

import collections
import warnings
import logging
import warnings
from typing import Union, List
import copy

from scipy.optimize import minpack

import astropy.units as u
import numpy
from astropy.convolution import Gaussian2DKernel
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky
from astropy.modeling import models, fitting
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.wcs.utils import pixel_to_skycoord
from astropy.wcs.utils import skycoord_to_pixel
from photutils import segmentation
from scipy import interpolate
from scipy.spatial.qhull import Voronoi

from rascil.data_models import Image, Skycomponent, get_parameter
from rascil.data_models.polarisation import PolarisationFrame, convert_pol_frame
from rascil.processing_components.calibration.jones import apply_jones
from rascil.processing_components.image.operations import create_image_from_array
from rascil.processing_components.skycomponent import copy_skycomponent
from rascil.processing_components.util.array_functions import (
    insert_function_sinc,
    insert_function_L,
    insert_function_pswf,
    insert_array,
)

log = logging.getLogger("rascil-logger")


def create_skycomponent(
    direction: SkyCoord,
    flux: numpy.array,
    frequency: numpy.array,
    shape: str = "Point",
    polarisation_frame=PolarisationFrame("stokesIQUV"),
    params: dict = None,
    name: str = "",
) -> Skycomponent:
    """A single Skycomponent with direction, flux, shape, and params for the shape

    :param direction: SkyCoord
    :param flux: [nchan, npol]
    :param frequency: [nchan]
    :param shape: 'Point' or 'Gaussian'
    :param polarisation_frame: e.g. PolarisationFrame("stokesIQUV")
    :param params:
    :param name: Name of component
    :return: Skycomponent
    """
    return Skycomponent(
        direction=direction,
        frequency=frequency,
        name=name,
        flux=numpy.array(flux),
        shape=shape,
        params=params,
        polarisation_frame=polarisation_frame,
    )


def find_nearest_skycomponent_index(home, comps) -> int:
    """Find nearest component in a list to a given direction (home)

    :param home: Home direction
    :param comps: list of skycomponents
    :return: index of best in comps
    """
    if len(comps) == 0:
        raise ValueError("find_nearest_skycomponent_index: Catalog is empty")
    catalog = SkyCoord(
        ra=[c.direction.ra for c in comps], dec=[c.direction.dec for c in comps]
    )
    idx, dist2d, dist3d = match_coordinates_sky(home, catalog)
    return idx


def find_nearest_skycomponent(home: SkyCoord, comps) -> (Skycomponent, float):
    """Find nearest component to a given direction

    :param home: Home direction
    :param comps: list of skycomponents
    :return: Index of nearest component
    """
    best_index = find_nearest_skycomponent_index(home, comps)
    best = comps[best_index]
    return best, best.direction.separation(home).rad


def find_separation_skycomponents(comps_test, comps_ref=None):
    """Find the matrix of separations for two lists of components

    :param comps_test: List of components to be test
    :param comps_ref: If None then set to comps_test
    :return:
    """
    if comps_ref is None:
        ncomps = len(comps_test)
        distances = numpy.zeros([ncomps, ncomps])
        for i in range(ncomps):
            for j in range(i + 1, ncomps):
                distances[i, j] = (
                    comps_test[i].direction.separation(comps_test[j].direction).rad
                )
                distances[j, i] = distances[i, j]
        return distances

    else:
        ncomps_ref = len(comps_ref)
        ncomps_test = len(comps_test)
        separations = numpy.zeros([ncomps_ref, ncomps_test])
        for ref in range(ncomps_ref):
            for test in range(ncomps_test):
                separations[ref, test] = (
                    comps_test[test].direction.separation(comps_ref[ref].direction).rad
                )

        return separations


def find_skycomponent_matches_atomic(comps_test, comps_ref, tol=1e-7):
    """Match a list of candidates to a reference set of skycomponents

    find_skycomponent_matches is faster since it uses the astropy catalog matching

    many to one is allowed.

    :param comps_test: skycomponents to test
    :param comps_ref: reference skycomponents
    :param tol: Tolerance in rad
    :return:
    """
    separations = find_separation_skycomponents(comps_test, comps_ref)
    matches = []
    for test, comp_test in enumerate(comps_test):
        best = numpy.argmin(separations[:, test])
        best_sep = separations[best, test]
        if best_sep < tol:
            matches.append((test, best, best_sep))

    assert len(matches) <= len(comps_test)

    return matches


def find_skycomponent_matches(comps_test, comps_ref, tol=1e-7):
    """Match a list of candidates to a reference set of skycomponents

    many to one is allowed.

    :param comps_test: skycomponents to test
    :param comps_ref: reference skycomponents
    :param tol: Tolerance in rad
    :return:
    """
    catalog_test = SkyCoord(
        ra=[c.direction.ra for c in comps_test],
        dec=[c.direction.dec for c in comps_test],
    )
    catalog_ref = SkyCoord(
        ra=[c.direction.ra for c in comps_ref], dec=[c.direction.dec for c in comps_ref]
    )
    idx, dist2d, dist3d = match_coordinates_sky(catalog_test, catalog_ref)
    matches = list()
    for test, comp_test in enumerate(comps_test):
        best = idx[test]
        best_sep = dist2d[test].rad
        if best_sep < tol:
            matches.append((test, best, best_sep))

    return matches


def select_components_by_separation(
    home, comps, rmax=2 * numpy.pi, rmin=0.0
) -> [Skycomponent]:
    """Select components with a range in separation

    :param home: Home direction
    :param comps: list of skycomponents
    :param rmin: minimum range
    :param rmax: maximum range
    :return: selected components
    """
    selected = list()
    for comp in comps:
        thissep = comp.direction.separation(home).rad
        if rmin <= thissep <= rmax:
            selected.append(comp)
    return selected


def select_neighbouring_components(comps, target_comps):
    """Assign components to nearest in the target

    :param comps: skycomponents
    :param target_comps: Target skycomponents
    :return: Indices of components in target_comps
    """
    target_catalog = SkyCoord(
        [c.direction.ra.rad for c in target_comps] * u.rad,
        [c.direction.dec.rad for c in target_comps] * u.rad,
    )

    all_catalog = SkyCoord(
        [c.direction.ra.rad for c in comps] * u.rad,
        [c.direction.dec.rad for c in comps] * u.rad,
    )

    from astropy.coordinates import match_coordinates_sky

    idx, d2d, d3d = match_coordinates_sky(all_catalog, target_catalog)
    return idx, d2d


def remove_neighbouring_components(comps, distance):
    """Remove the faintest of a pair of components that are within a specified distance

    :param comps: skycomponents
    :param distance: Minimum distance
    :return: Indices of components in target_comps, selected components
    """
    ncomps = len(comps)
    ok = ncomps * [True]
    for i in range(ncomps):
        if ok[i]:
            for j in range(i + 1, ncomps):
                if ok[j]:
                    d = comps[i].direction.separation(comps[j].direction).rad
                    if d < distance:
                        if numpy.max(comps[i].flux) > numpy.max(comps[j].flux):
                            ok[j] = False
                        else:
                            ok[i] = False
                        break

    from itertools import compress

    idx = list(compress(list(range(ncomps)), ok))
    comps_sel = list(compress(comps, ok))
    return idx, comps_sel


def find_skycomponents(
    im: Image, fwhm=1.0, threshold=1.0, npixels=5
) -> List[Skycomponent]:
    """Find gaussian components in Image above a certain threshold as Skycomponent

    :param im: Image to be searched
    :param fwhm: Full width half maximum of gaussian in pixels
    :param threshold: Threshold for component detection. Default: 1 Jy.
    :param npixels: Number of connected pixels required
    :return: list of sky components
    """

    # assert isinstance(im, Image)
    log.debug("find_skycomponents: Finding components in Image by segmentation")

    # We use photutils segmentation - this first segments the image
    # into pieces that are thought to contain individual sources, then
    # identifies the concrete source properties. Having these two
    # steps makes it straightforward to extract polarisation and
    # spectral information.

    # Make filter kernel
    sigma = fwhm * gaussian_fwhm_to_sigma
    kernel = Gaussian2DKernel(sigma, x_size=int(1.5 * fwhm), y_size=int(1.5 * fwhm))
    kernel.normalize()

    # Segment the average over all channels of Stokes I
    image_sum = numpy.sum(im["pixels"].data, axis=0)[0, ...] / float(
        im["pixels"].data.shape[0]
    )
    segments = segmentation.detect_sources(
        image_sum, threshold, npixels=npixels, filter_kernel=kernel
    )
    assert segments is not None, "Failed to find any components"

    log.info("find_skycomponents: Identified %d segments" % segments.nlabels)

    # Now compute source properties for all polarisations and frequencies
    comp_tbl = [
        [
            segmentation.source_properties(
                im["pixels"].data[chan, pol],
                segments,
                filter_kernel=kernel,
            ).to_table()
            for pol in [0]
        ]
        for chan in range(im.image_acc.nchan)
    ]

    def comp_prop(comp, prop_name):
        return [
            [comp_tbl[chan][pol][comp][prop_name] for pol in [0]]
            for chan in range(im.image_acc.nchan)
        ]

    # Generate components
    comps = []
    for segment in range(segments.nlabels):
        # Get flux and position. Astropy's quantities make this
        # unnecessarily complicated.
        flux = numpy.array(comp_prop(segment, "max_value"))
        xs = u.Quantity(list(map(u.Quantity, comp_prop(segment, "maxval_xpos"))))
        ys = u.Quantity(list(map(u.Quantity, comp_prop(segment, "maxval_ypos"))))

        sc = pixel_to_skycoord(xs, ys, im.image_acc.wcs, 0)
        ras = sc.ra
        decs = sc.dec

        # Determine "true" position by weighting
        aflux = numpy.abs(flux)
        flux_sum = numpy.sum(aflux)
        ra = numpy.sum(aflux * ras) / flux_sum
        dec = numpy.sum(aflux * decs) / flux_sum
        xs = numpy.sum(aflux * xs) / flux_sum
        ys = numpy.sum(aflux * ys) / flux_sum

        point_flux = im["pixels"].data[
            :,
            :,
            numpy.round(ys.value).astype("int"),
            numpy.round(xs.value).astype("int"),
        ]

        # Add component
        comps.append(
            Skycomponent(
                direction=SkyCoord(ra=ra, dec=dec),
                frequency=im.frequency,
                name="Segment %d" % segment,
                flux=point_flux,
                shape="Point",
                polarisation_frame=im.image_acc.polarisation_frame,
                params={},
            )
        )

    return comps


def apply_beam_to_skycomponent(
    sc: Union[Skycomponent, List[Skycomponent]],
    beam: Image,
    phasecentre=None,
    inverse=False,
) -> Union[Skycomponent, List[Skycomponent]]:
    """Apply a primary beam to a Skycomponent

    if inverse==True, do an inverse where we subtract the primary beam from the skycomponents
    if inverse==False, do a multiplication of beam and skycomponent fluxes
    :param phasecentre:
    :param beam: primary beam
    :param sc: SkyComponent or list of SkyComponents
    :return: List of skycomponents
    """
    ##assert isinstance(beam, Image)
    single = not isinstance(sc, collections.abc.Iterable)

    if single:
        sc = [sc]

    nchan, npol, ny, nx = beam["pixels"].data.shape

    log.debug("apply_beam_to_skycomponent: Processing %d components" % (len(sc)))

    ras = [comp.direction.ra.radian for comp in sc]
    decs = [comp.direction.dec.radian for comp in sc]
    skycoords = SkyCoord(ras * u.rad, decs * u.rad, frame="icrs")
    if beam.image_acc.wcs.wcs.ctype[0] == "RA---SIN":
        pixlocs = skycoord_to_pixel(skycoords, beam.image_acc.wcs, origin=1, mode="wcs")
    else:
        wcs = copy.deepcopy(beam.image_acc.wcs)
        wcs.wcs.ctype[0] = "RA---SIN"
        wcs.wcs.ctype[1] = "DEC--SIN"
        wcs.wcs.crval[0] = phasecentre.ra.deg
        wcs.wcs.crval[1] = phasecentre.dec.deg
        pixlocs = skycoord_to_pixel(skycoords, wcs, origin=1, mode="wcs")

    newsc = []
    total_flux = numpy.zeros_like(sc[0].flux)
    for icomp, comp in enumerate(sc):

        assert comp.shape == "Point", "Cannot handle shape %s" % comp.shape

        pixloc = (pixlocs[0][icomp], pixlocs[1][icomp])
        if not numpy.isnan(pixloc).any():
            x, y = int(round(float(pixloc[0]))), int(round(float(pixloc[1])))
            if 0 <= x < nx and 0 <= y < ny:
                if inverse and beam["pixels"].data[:, :, y, x] is not 0.0:
                    comp_flux = comp.flux / beam["pixels"].data[:, :, y, x]
                else:
                    comp_flux = comp.flux * beam["pixels"].data[:, :, y, x]
                total_flux += comp_flux
            else:
                comp_flux = 0.0 * comp.flux
            newsc.append(
                Skycomponent(
                    comp.direction,
                    comp.frequency,
                    comp.name,
                    comp_flux,
                    shape=comp.shape,
                    polarisation_frame=comp.polarisation_frame,
                )
            )

    log.debug(
        "apply_beam_to_skycomponent: %d components with total flux %s"
        % (len(newsc), total_flux)
    )
    if single:
        return newsc[0]
    else:
        return newsc


def apply_voltage_pattern_to_skycomponent(
    sc: Union[Skycomponent, List[Skycomponent]],
    vp: Image,
    inverse=False,
    phasecentre=None,
) -> Union[Skycomponent, List[Skycomponent]]:
    """Apply a voltage pattern to a Skycomponent

    For inverse==False, input polarisation_frame must be stokesIQUV, and
    output polarisation_frame is same as voltage pattern

    For inverse==True, input polarisation_frame must be same as voltage pattern, and
    output polarisation_frame is "stokesIQUV"

    Requires a complex Image with the correct ordering of polarisation axes:
    e.g. RR, LL, RL, LR or XX, YY, XY, YX

    :param inverse:
    :param vp: voltage pattern as complex image
    :param sc: SkyComponent or list of SkyComponents
    :return: List of skycomponents
    """
    # assert isinstance(vp, Image)
    assert (vp.image_acc.polarisation_frame == PolarisationFrame("linear")) or (
        vp.image_acc.polarisation_frame == PolarisationFrame("circular")
    )

    # assert vp["pixels"].data.dtype == "complex128"
    single = not isinstance(sc, collections.abc.Iterable)

    if single:
        sc = [sc]

    nchan, npol, ny, nx = vp["pixels"].data.shape

    log.debug("apply_vp_to_skycomponent: Processing %d components" % (len(sc)))

    ras = [comp.direction.ra.radian for comp in sc]
    decs = [comp.direction.dec.radian for comp in sc]
    skycoords = SkyCoord(ras * u.rad, decs * u.rad, frame="icrs")
    if vp.image_acc.wcs.wcs.ctype[0] == "RA---SIN":
        pixlocs = skycoord_to_pixel(skycoords, vp.image_acc.wcs, origin=1, mode="wcs")
    else:
        assert phasecentre is not None, "Need to know the phasecentre"
        wcs = copy.deepcopy(vp.image_acc.wcs)
        wcs.wcs.ctype[0] = "RA---SIN"
        wcs.wcs.ctype[1] = "DEC--SIN"
        wcs.wcs.crval[0] = phasecentre.ra.deg
        wcs.wcs.crval[1] = phasecentre.dec.deg
        pixlocs = skycoord_to_pixel(skycoords, wcs, origin=1, mode="wcs")

    newsc = []
    total_flux = numpy.zeros([nchan, npol], dtype="complex")

    for icomp, comp in enumerate(sc):

        assert comp.shape == "Point", "Cannot handle shape %s" % comp.shape

        # Convert to linear (xx, xy, yx, yy) or circular (rr, rl, lr, ll)
        nchan, npol = comp.flux.shape
        assert npol == 4
        if not inverse:
            assert comp.polarisation_frame == PolarisationFrame("stokesIQUV")

        comp_flux_cstokes = convert_pol_frame(
            comp.flux, comp.polarisation_frame, vp.image_acc.polarisation_frame
        ).reshape([nchan, 2, 2])
        comp_flux = numpy.zeros([nchan, npol], dtype="complex")

        pixloc = (pixlocs[0][icomp], pixlocs[1][icomp])
        if not numpy.isnan(pixloc).any():
            x, y = int(round(float(pixloc[0]))), int(round(float(pixloc[1])))
            if 0 <= x < nx and 0 <= y < ny:
                # Now we want to left and right multiply by the Jones matrices
                # comp_flux = vp["pixels"].data[:, :, y, x] * comp_flux_cstokes * numpy.vp["pixels"].data[:, :, y, x]
                for chan in range(nchan):
                    ej = vp["pixels"].data[chan, :, y, x].reshape([2, 2])
                    cfs = comp_flux_cstokes[chan].reshape([2, 2])
                    comp_flux[chan, :] = apply_jones(ej, cfs, inverse).reshape([4])

                total_flux += comp_flux
                if inverse:
                    comp_flux = convert_pol_frame(
                        comp_flux,
                        vp.image_acc.polarisation_frame,
                        PolarisationFrame("stokesIQUV"),
                    )
                    comp.polarisation_frame = PolarisationFrame("stokesIQUV")

                newsc.append(
                    Skycomponent(
                        comp.direction,
                        comp.frequency,
                        comp.name,
                        comp_flux,
                        shape=comp.shape,
                        polarisation_frame=vp.image_acc.polarisation_frame,
                    )
                )

    log.debug(
        "apply_vp_to_skycomponent: %d components with total flux %s"
        % (len(newsc), total_flux)
    )
    if single:
        return newsc[0]
    else:
        return newsc


def filter_skycomponents_by_index(sc, indices):
    """Filter sky components by index

    :param sc: List of Skycomponents
    :param indices: Allowed indecies
    :return:
    """
    newcomps = list()
    for i in indices:
        newcomps.append(sc[i])

    return newcomps


def filter_skycomponents_by_flux(sc, flux_min=-numpy.inf, flux_max=numpy.inf):
    """Filter sky components by stokes I flux

    :param sc: List of Skycomponents
    :param flux_min: Minimum I flux
    :param flux_max: Maximum I flux
    :return:
    """
    newcomps = list()
    for comp in sc:
        if (numpy.max(comp.flux[:, 0]) > flux_min) and (
            numpy.max(comp.flux[:, 0]) < flux_max
        ):
            newcomps.append(comp)

    return newcomps


def insert_skycomponent(
    im: Image,
    sc: Union[Skycomponent, List[Skycomponent]],
    insert_method="Nearest",
    bandwidth=1.0,
    support=8,
) -> Image:
    """Insert a Skycomponent into an image

    :param im: Image
    :param sc: SkyComponent or list of SkyComponents
    :param insert_method: '' | 'Sinc' | 'Lanczos'
    :param bandwidth: Fractional of uv plane to optimise over (1.0)
    :param support: Support of kernel (7)
    :return: Image
    """

    ##assert isinstance(im, Image)

    support = int(support / bandwidth)

    nchan, npol, ny, nx = im["pixels"].data.shape

    if not isinstance(sc, collections.abc.Iterable):
        sc = [sc]

    log.debug("insert_skycomponent: Using insert method %s" % insert_method)

    image_frequency = im.frequency.data

    ras = [comp.direction.ra.radian for comp in sc]
    decs = [comp.direction.dec.radian for comp in sc]
    skycoords = SkyCoord(ras * u.rad, decs * u.rad, frame="icrs")
    pixlocs = skycoord_to_pixel(skycoords, im.image_acc.wcs, origin=0, mode="wcs")

    nbad = 0
    for icomp, comp in enumerate(sc):

        assert comp.shape == "Point", "Cannot handle shape %s" % comp.shape

        pixloc = (pixlocs[0][icomp], pixlocs[1][icomp])
        flux = numpy.zeros([nchan, npol])

        if (
            len(comp.frequency.data) == len(image_frequency)
            and numpy.max(numpy.abs(comp.frequency.data - image_frequency)) < 1e-7
        ):
            flux = comp.flux
        elif comp.flux.shape[0] > 1:
            for pol in range(npol):
                fint = interpolate.interp1d(
                    comp.frequency.data, comp.flux[:, pol], kind="cubic"
                )
                flux[:, pol] = fint(image_frequency)
        else:
            flux = comp.flux

        if insert_method == "Lanczos":
            insert_array(
                im["pixels"].data,
                pixloc[0],
                pixloc[1],
                flux,
                bandwidth,
                support,
                insert_function=insert_function_L,
            )
        elif insert_method == "Sinc":
            insert_array(
                im["pixels"].data,
                pixloc[0],
                pixloc[1],
                flux,
                bandwidth,
                support,
                insert_function=insert_function_sinc,
            )
        elif insert_method == "PSWF":
            insert_array(
                im["pixels"].data,
                pixloc[0],
                pixloc[1],
                flux,
                bandwidth,
                support,
                insert_function=insert_function_pswf,
            )
        else:
            insert_method = "Nearest"
            y, x = numpy.round(pixloc[1]).astype("int"), numpy.round(pixloc[0]).astype(
                "int"
            )
            if 0 <= x < nx and 0 <= y < ny:
                im["pixels"].data[:, :, y, x] += flux[...]
            else:
                nbad += 1
    if nbad > 0:
        log.warning(
            f"insert_skycomponent: {nbad} components of {len(sc)} do not fit on image"
        )

    return im


def restore_skycomponent(
    im: Image,
    sc: Union[Skycomponent, List[Skycomponent]],
    clean_beam=None,
    support=8,
) -> Image:
    """Restore a Skycomponent into an image

    :param im: Image
    :param sc: SkyComponent or list of SkyComponents
    :param clean_beam: dict e.g. {"bmaj":0.1, "bmin":0.05, "bpa":-60.0}. Units are deg, deg, deg
    :param support: Support of kernel (7)
    :return: Image
    """

    nchan, npol, ny, nx = im["pixels"].data.shape

    if not isinstance(sc, collections.abc.Iterable):
        sc = [sc]

    image_frequency = im.frequency.data

    ras = [comp.direction.ra.radian for comp in sc]
    decs = [comp.direction.dec.radian for comp in sc]
    skycoords = SkyCoord(ras * u.rad, decs * u.rad, frame="icrs")
    pixlocs = skycoord_to_pixel(skycoords, im.image_acc.wcs, origin=0, mode="wcs")

    from rascil.processing_components.image.operations import (
        convert_clean_beam_to_pixels,
    )

    beam_pixels = convert_clean_beam_to_pixels(im, clean_beam)

    for icomp, comp in enumerate(sc):

        if comp.shape != "Point":
            raise ValueError(
                "restore_skycomponent: Cannot handle shape %s" % comp.shape
            )

        pixloc = (pixlocs[0][icomp], pixlocs[1][icomp])
        flux = numpy.zeros([nchan, npol])

        if (
            len(comp.frequency.data) == len(image_frequency)
            and numpy.max(numpy.abs(comp.frequency.data - image_frequency)) < 1e-7
        ):
            flux = comp.flux
        elif comp.flux.shape[0] > 1:
            for pol in range(npol):
                fint = interpolate.interp1d(
                    comp.frequency.data, comp.flux[:, pol], kind="cubic"
                )
                flux[:, pol] = fint(image_frequency)
        else:
            flux = comp.flux

        gaussian = models.Gaussian2D(
            amplitude=1.0,
            x_mean=pixloc[0],
            y_mean=pixloc[1],
            x_stddev=beam_pixels[0],
            y_stddev=beam_pixels[1],
            theta=beam_pixels[2],
        )
        xi, yi = numpy.indices(im["pixels"].data.shape[-2:])
        im["pixels"].data[...] += (
            flux[..., numpy.newaxis, numpy.newaxis]
            * gaussian(yi, xi)[numpy.newaxis, numpy.newaxis, ...]
        )

    im.attrs["clean_beam"] = clean_beam
    return im


def voronoi_decomposition(im, comps):
    """Construct a Voronoi decomposition of a set of components

    The array return contains the index into the scipy.spatial.qhull.Voronoi structure

    :param im: Image
    :param comps: List of Skycomponents
    :return: Voronoi structure, vertex image
    """

    def voronoi_vertex(vy, vx, vertex_y, vertex_x):
        """Return the nearest Voronoi vertex

        :param vy:
        :param vx:
        :param vertex_y:
        :param vertex_x:
        :return:
        """
        return numpy.argmin(numpy.hypot(vy - vertex_y, vx - vertex_x))

    directions = SkyCoord(
        [u.rad * c.direction.ra.rad for c in comps],
        [u.rad * c.direction.dec.rad for c in comps],
    )
    x, y = skycoord_to_pixel(directions, im.image_acc.wcs, 1, "wcs")
    points = [(x[i], y[i]) for i, _ in enumerate(x)]
    vor = Voronoi(points)

    nchan, npol, ny, nx = im["pixels"].data.shape
    vertex_image = numpy.zeros([ny, nx]).astype("int")
    for j in range(ny):
        for i in range(nx):
            vertex_image[j, i] = voronoi_vertex(
                j, i, vor.points[:, 1], vor.points[:, 0]
            )

    return vor, vertex_image


def image_voronoi_iter(im: Image, components: list) -> collections.abc.Iterable:
    """Iterate through Voronoi decomposition, returning a generator yielding fullsize images

    :param im: Image
    :param components: Components to define Voronoi decomposition
    :returns: generator of Images
    """
    if len(components) == 1:
        mask = numpy.ones(im["pixels"].data.shape)
        yield create_image_from_array(
            mask,
            wcs=im.image_acc.wcs,
            polarisation_frame=im.image_acc.polarisation_frame,
        )
    else:
        vor, vertex_array = voronoi_decomposition(im, components)

        nregions = numpy.max(vertex_array) + 1
        for region in range(nregions):
            mask = numpy.zeros(im["pixels"].data.shape)
            mask[(vertex_array == region)[numpy.newaxis, numpy.newaxis, ...]] = 1.0
            yield create_image_from_array(
                mask,
                wcs=im.image_acc.wcs,
                polarisation_frame=im.image_acc.polarisation_frame,
            )


def partition_skycomponent_neighbours(comps, targets):
    """Partition sky components by nearest target source

    :param comps:
    :param targets:
    :return:
    """
    idx, d2d = select_neighbouring_components(comps, targets)

    from itertools import compress

    comps_lists = list()
    for comp_id in numpy.unique(idx):
        selected_comps = list(compress(comps, idx == comp_id))
        comps_lists.append(selected_comps)

    return comps_lists


def fit_skycomponent(im: Image, sc: Skycomponent, **kwargs):
    """Fit a two dimensional Gaussian skycomponent using astropy.modeling

    :params im: Input image
    :params sc: Skycomponent
    :return: Skycomponent
    """
    pixloc = numpy.round(
        skycoord_to_pixel(sc.direction, im.image_acc.wcs, origin=0)
    ).astype("int")
    sl_y = slice(pixloc[1] - 7, pixloc[1] + 8)
    sl_x = slice(pixloc[0] - 7, pixloc[0] + 8)

    # im["pixels"].data[0,0,150,121]
    y, x = numpy.mgrid[sl_y, sl_x]
    z = im["pixels"].data[0, 0, sl_y, sl_x]

    image_shape = im["pixels"].data[0, 0].shape
    # isotropic at the moment!

    newsc = copy_skycomponent(sc)

    try:
        p_init = models.Gaussian2D(
            amplitude=numpy.max(z), x_mean=numpy.mean(x), y_mean=numpy.mean(y)
        )

        fit_p = fitting.LevMarLSQFitter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = fit_p(p_init, x, y, z)

        # Now fill in the new skycomponent values
        newsc.direction = pixel_to_skycoord(fit.x_mean, fit.y_mean, im.image_acc.wcs, 0)
        iy = round(fit.y_mean.value)
        ix = round(fit.x_mean.value)

        # We could fit each frequency separately. For the moment, we just scale
        if iy < image_shape[0] and iy >= 0 and ix < image_shape[1] and ix >= 0:
            newsc.flux = im["pixels"].data[:, :, iy, ix]

        force_point_sources = get_parameter(kwargs, "force_point_sources", True)
        if force_point_sources or (fit.x_fwhm <= 0.0 or fit.y_fwhm <= 0.0):
            newsc.shape = "Point"
        else:
            newsc.shape = "Gaussian"
            # cellsize in radians
            cellsize = numpy.abs((im["x"][0].data - im["x"][-1].data)) / len(im["x"])

            gaussian_pixels = (fit.x_fwhm, fit.y_fwhm, fit.theta)

            if gaussian_pixels[1] > gaussian_pixels[0]:
                clean_gaussian = {
                    "bmaj": numpy.rad2deg(gaussian_pixels[1] * cellsize),
                    "bmin": numpy.rad2deg(gaussian_pixels[0] * cellsize),
                    "bpa": numpy.rad2deg(gaussian_pixels[2]),
                }
            else:
                clean_gaussian = {
                    "bmaj": numpy.rad2deg(gaussian_pixels[0] * cellsize),
                    "bmin": numpy.rad2deg(gaussian_pixels[1] * cellsize),
                    "bpa": numpy.rad2deg(gaussian_pixels[2]) + 90.0,
                }
            newsc.shape = "Gaussian"
            newsc.params = clean_gaussian

    except (minpack.error, ValueError) as err:
        log.warning(f"fit_skycomponent: fit failed {err}")
        return sc

    return newsc


def fit_skycomponent_spectral_index(sc: Skycomponent):

    """Fit the spectral index for a multi frequency skycomponent

    :param sc: Skycomponent
    :return: spec_indx
    """
    # Function after taken log
    power_law_func = lambda a, b, x: a * x + b

    nchan = sc.frequency.shape[0]

    if nchan <= 1:
        log.warning("Single frequency skycomponent, skip fitting")
        spec_indx = 0.0

    else:
        centre = nchan // 2
        if sc.frequency[centre] > 0.0 and sc.flux[centre, 0] > 0.0:
            xdata = numpy.log10(sc.frequency / sc.frequency[centre])
            ydata = numpy.log10(sc.flux[:, 0] / sc.flux[centre, 0])
            out = numpy.polyfit(xdata, ydata, 1)
            spec_indx = out[0]
        else:
            log.warning("Wrong values encountered, no fitting performed.")
            spec_indx = 0.0

    return spec_indx
