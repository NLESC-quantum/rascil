"""
Functions that define and manipulate images. Images are just data and a World Coordinate System.
"""

__all__ = ["image_channel_iter", "image_null_iter", "image_raster_iter"]

import logging
import collections.abc

import numpy

from rascil.data_models.memory_data_models import Image

from rascil.processing_components.image.operations import (
    create_image_from_array,
    create_empty_image_like,
    image_is_canonical,
)
from rascil.processing_components.util.array_functions import tukey_filter

log = logging.getLogger("rascil-logger")


def image_null_iter(im: Image, facets=1, overlap=0) -> collections.abc.Iterable:
    """One time iterator

    This is useful to simplify control structures.

    :param im:
    :param facets: Number of image partitions on each axis (2)
    :param overlap: overlap in pixels
    :return:
    """
    yield im


def image_raster_iter(
    im: Image, facets=1, overlap=0, taper="flat", make_flat=False
) -> collections.abc.Iterable:
    """Create an image_raster_iter generator, returning a list of subimages, optionally with overlaps

    The WCS is adjusted appropriately for each raster element. Hence this is a coordinate-aware
    way to iterate through an image.

    Provided we don't break reference semantics, memory should be conserved. However make_flat
    creates a new set of images and thus reference semantics dont hold.

    To update the image in place::

        for r in image_raster_iter(im, facets=2):
            r["pixels'].data[...] = numpy.sqrt(r["pixels'].data[...])

    If the overlap is greater than zero, we choose to keep all images the same size so the
    other ring of facets are ignored. So if facets=4 and overlap > 0 then the iterator returns
    (facets-2)**2 = 4 images.

    A taper is applied in the overlap regions. None implies a constant value, linear is a ramp, and
    quadratic is parabolic at the ends.

    :param im: Image
    :param facets: Number of image partitions on each axis (2)
    :param overlap: overlap in pixels
    :param taper: method of tapering at the edges: 'flat' or 'linear' or 'quadratic' or 'tukey'
    :param make_flat: Make the flat images
    :returns: Generator of images

    See also
        :py:func:`rascil.processing_components.image.gather_scatter.image_gather_facets`
        :py:func:`rascil.processing_components.image.gather_scatter.image_scatter_facets`
    """

    assert image_is_canonical(im)
    
    if im is None:
        return im

    nchan, npol, ny, nx = im["pixels"].data.shape
    assert facets <= ny, "Cannot have more raster elements than pixels"
    assert facets <= nx, "Cannot have more raster elements than pixels"

    assert facets >= 1, "Facets cannot be zero or less"
    assert overlap >= 0, "Overlap must be zero or greater"

    if facets == 1:
        yield im
    else:

        if overlap >= (nx // facets) or overlap >= (ny // facets):
            raise ValueError(f"Overlap in facets is too large {nx}, {facets}, {overlap}")

        # Size of facet
        dx = nx // facets
        dy = nx // facets

        # Step between facets
        sx = nx // facets - 2 * overlap
        sy = ny // facets - 2 * overlap

        def taper_linear():
            t = numpy.ones(dx)
            ramp = numpy.arange(0, overlap).astype(float) / float(overlap)

            t[:overlap] = ramp
            t[(dx - overlap) : dx] = 1.0 - ramp
            result = numpy.outer(t, t)

            return result

        def taper_quadratic():
            t = numpy.ones(dx)
            ramp = numpy.arange(0, overlap).astype(float) / float(overlap)

            quadratic_ramp = numpy.ones(overlap)
            quadratic_ramp[0 : overlap // 2] = 2.0 * ramp[0 : overlap // 2] ** 2
            quadratic_ramp[overlap // 2 :] = 1 - 2.0 * ramp[overlap // 2 : 0 : -1] ** 2

            t[:overlap] = quadratic_ramp
            t[(dx - overlap) : dx] = 1.0 - quadratic_ramp

            result = numpy.outer(t, t)
            return result

        def taper_tukey():

            xs = numpy.arange(dx) / float(dx)
            r = 2 * overlap / dx
            t = [tukey_filter(x, r) for x in xs]

            result = numpy.outer(t, t)
            return result

        def taper_flat():
            return numpy.ones([dx, dx])

        i = 0
        for fy in range(facets):
            y = ny // 2 + sy * (fy - facets // 2) - overlap
            for fx in range(facets):
                x = nx // 2 + sx * (fx - facets // 2) - overlap
                if x < 0 or x + dx > nx:
                    raise ValueError(f"overlap too large: starting point {x}")
                wcs = im.image_acc.wcs.deepcopy()
                wcs.wcs.crpix[0] -= x
                wcs.wcs.crpix[1] -= y
                # yield image from slice (reference!)
                subim = create_image_from_array(
                    im["pixels"].data[..., y : y + dy, x : x + dx],
                    wcs,
                    im.image_acc.polarisation_frame,
                )
                if overlap > 0 and make_flat:
                    flat = create_empty_image_like(subim)
                    if taper == "linear":
                        flat["pixels"].data[..., :, :] = taper_linear()
                    elif taper == "quadratic":
                        flat["pixels"].data[..., :, :] = taper_quadratic()
                    elif taper == "tukey":
                        flat["pixels"].data[..., :, :] = taper_tukey()
                    else:
                        flat["pixels"].data[..., :, :] = taper_flat()
                    yield flat
                else:
                    yield subim
                i += 1


def image_channel_iter(im: Image, subimages=1) -> collections.abc.Iterable:
    """Create a image_channel_iter generator, returning images

    The WCS is adjusted appropriately for each raster element. Hence this is a coordinate-aware
    way to iterate through an image.

    Provided we don't break reference semantics, memory should be conserved

    To update the image in place::

        for r in image_channel_iter(im, subimages=nchan):
            r.data[...] = numpy.sqrt(r.data[...])

    :param im: Image
    :param subimages: Number of subimages
    :returns: Generator of images

    See also
        :py:func:`rascil.processing_components.image.gather_scatter.image_gather_channels`
        :py:func:`rascil.processing_components.image.gather_scatter.image_scatter_channels`
    """

    assert image_is_canonical(im)

    nchan, npol, ny, nx = im["pixels"].data.shape

    assert subimages <= nchan, "More subimages %d than channels %d" % (subimages, nchan)
    step = nchan // subimages
    channels = numpy.array(range(0, nchan, step), dtype="int")
    assert (
        len(channels) == subimages
    ), "subimages %d does not match length of channels %d" % (subimages, len(channels))

    for i, channel in enumerate(channels):
        if i + 1 < len(channels):
            channel_max = channels[i + 1]
        else:
            channel_max = nchan

        # Adjust WCS
        wcs = im.image_acc.wcs.deepcopy()
        wcs.wcs.crpix[3] -= channel

        # Yield image from slice (reference!)
        yield create_image_from_array(
            im["pixels"].data[channel:channel_max, ...],
            wcs,
            im.image_acc.polarisation_frame,
        )
