#
"""
Functions that perform gather/scatter operations on Images.
"""

__all__ = [
    "image_gather_channels",
    "image_scatter_channels",
    "image_gather_facets",
    "image_scatter_facets",
]

import copy
import logging
from typing import List

import xarray
import numpy

from rascil.data_models.memory_data_models import Image
from rascil.processing_components.image.iterators import (
    image_raster_iter,
    image_channel_iter,
)
from rascil.processing_components.image.operations import (
    create_empty_image_like,
    create_image_from_array,
    image_is_canonical,
)

log = logging.getLogger("rascil-logger")


def image_scatter_facets(im: Image, facets=1, overlap=0, taper=None) -> List[Image]:
    """Scatter an image into a list of subimages using the  image_raster_iterator

    If the overlap is greater than zero, we choose to keep all images the same size so the
    other ring of facets are ignored. So if facets=4 and overlap > 0 then the scatter returns
    (facets-2)**2 = 4 images.

    :param im: Image
    :param facets: Number of image partitions on each axis (2)
    :param overlap: Number of pixels overlap
    :param taper: Taper at edges None or 'linear'
    :return: list of subimages

    See also:
        :py:func:`processing_components.image.iterators.image_raster_iter`
    """
    return [
        flat_facet
        for flat_facet in image_raster_iter(
            im, facets=facets, overlap=overlap, taper=taper
        )
    ]


def image_gather_facets(
    image_list: List[Image],
    im: Image,
    facets=1,
    overlap=0,
    taper=None,
    return_flat=False,
):
    """Gather a list of subimages back into an image using the  image_raster_iterator

    If the overlap is greater than zero, we choose to keep all images the same size so the
    other ring of facets are ignored. So if facets=4 and overlap > 0 then the gather expects
    (facets-2)**2 = 4 images.

    To normalize the overlap we make a set of flats, gather that and divide. The flat may be optionally returned
    instead of the result

    :param image_list: List of subimages
    :param im: Output image
    :param facets: Number of image partitions on each axis (2)
    :param overlap: Overlap between neighbours in pixels
    :param taper: Taper at edges None or 'linear' or 'Tukey'
    :param return_flat: Return the flat
    :return: list of subimages

    See also
        :py:func:`rascil.processing_components.image.iterators.image_raster_iter`
    """
    out = create_empty_image_like(im)
    if overlap > 0:
        flat = create_empty_image_like(im)
        flat["pixels"].data[...] = 1.0
        flats = [
            f
            for f in image_raster_iter(
                flat, facets=facets, overlap=overlap, taper=taper, make_flat=True
            )
        ]

        sum_flats = create_empty_image_like(im)

        if return_flat:
            i = 0
            for sum_flat_facet in image_raster_iter(
                sum_flats, facets=facets, overlap=overlap, taper=taper
            ):
                sum_flat_facet["pixels"].data[...] += flats[i]["pixels"].data[...]
                i += 1

            return sum_flats
        else:
            i = 0
            for out_facet, sum_flat_facet in zip(
                image_raster_iter(out, facets=facets, overlap=overlap, taper=taper),
                image_raster_iter(
                    sum_flats, facets=facets, overlap=overlap, taper=taper
                ),
            ):
                out_facet["pixels"].data[...] += (
                    flats[i]["pixels"].data * image_list[i]["pixels"].data[...]
                )
                sum_flat_facet["pixels"].data[...] += flats[i]["pixels"].data[...]
                i += 1

            out["pixels"].data[sum_flats["pixels"].data > 0.0] /= sum_flats[
                "pixels"
            ].data[sum_flats["pixels"].data > 0.0]
            out["pixels"].data[sum_flats["pixels"].data <= 0.0] = 0.0

            return out
    else:
        flat = create_empty_image_like(im)
        flat["pixels"].data[...] = 1.0

        if return_flat:
            return flat
        else:
            for i, facet in enumerate(
                image_raster_iter(out, facets=facets, overlap=overlap, taper=taper)
            ):
                facet["pixels"].data[...] += image_list[i]["pixels"].data[...]

            return out


def image_scatter(im: Image, dim="frequency", subimages=None) -> List[Image]:
    """Scatter an image into a list of subimages using the dimension dim

    :param im: Image
    :param subimages: Number of channels
    :return: list of subimages

    See also
        :py:func:`rascil.processing_components.image.iterators.image_channel_iter`
    """
    return [r[1] for r in im.groupby_bins(dim, bins=subimages, squeeze=False)]


def image_scatter_channels(im: Image, subimages=None) -> List[Image]:
    """Scatter an image into a list of subimages using the channels

    :param im: Image
    :param subimages: Number of channels
    :return: list of subimages

    See also
        :py:func:`rascil.processing_components.image.iterators.image_channel_iter`
    """
    return image_scatter(im, "frequency")


def image_gather_channels(
    image_list: List[Image], im: Image = None, subimages=0
) -> Image:
    """Gather a list of subimages back into an image using the channel_iterator

    If the template image is not given then it will be formed assuming that the list has
    been generated by image_scatter_channels with subimages = number of channels

    :param image_list: List of subimages
    :param im: Output image
    :param subimages: Number of image partitions on each axis (2)
    :return: list of subimages
    """
    return xarray.concat(image_list, dim="frequency")
