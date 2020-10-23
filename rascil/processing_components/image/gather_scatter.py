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

import logging
from typing import List

import numpy

from rascil.data_models.memory_data_models import Image

from rascil.processing_components.image.operations import (
    create_image_from_array,
    create_empty_image_like,
    image_is_canonical,
)
from rascil.processing_components.image.iterators import (
    image_raster_iter,
    image_channel_iter,
)

log = logging.getLogger("logger")


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
        flat.data[...] = 1.0
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
                sum_flat_facet.data[...] += flats[i].data[...]
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
                out_facet.data[...] += flats[i].data * image_list[i].data[...]
                sum_flat_facet.data[...] += flats[i].data[...]
                i += 1

            out.data[sum_flats.data > 0.0] /= sum_flats.data[sum_flats.data > 0.0]
            out.data[sum_flats.data <= 0.0] = 0.0

            return out
    else:
        flat = create_empty_image_like(im)
        flat.data[...] = 1.0

        if return_flat:
            return flat
        else:
            for i, facet in enumerate(
                image_raster_iter(out, facets=facets, overlap=overlap, taper=taper)
            ):
                facet.data[...] += image_list[i].data[...]

            return out


def image_scatter_channels(im: Image, subimages=None) -> List[Image]:
    """Scatter an image into a list of subimages using the channels

    :param im: Image
    :param subimages: Number of channels
    :return: list of subimages

    See also
        :py:func:`rascil.processing_components.image.iterators.image_channel_iter`
    """

    assert image_is_canonical(im)

    image_list = list()
    if subimages is None:
        subimages = im.shape[0]

    for slab in image_channel_iter(im, subimages=subimages):
        image_list.append(slab)

    assert len(image_list) == subimages, "Too many subimages scattered"

    return image_list


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

    if im is None:
        nchan = len(image_list)
        _, npol, ny, nx = image_list[0].shape
        im_shape = nchan, npol, ny, ny
        im = create_image_from_array(
            numpy.zeros(im_shape, dtype=image_list[0].data.dtype),
            image_list[0].wcs,
            image_list[0].polarisation_frame,
        )

    assert image_is_canonical(im)

    if subimages == 0:
        subimages = len(image_list)

    for i, slab in enumerate(image_channel_iter(im, subimages=subimages)):
        slab.data[...] = image_list[i].data[...]

    return im
