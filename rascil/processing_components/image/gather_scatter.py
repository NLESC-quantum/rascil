#
"""
Functions that perform gather/scatter operations on Images.
"""

__all__ = ['image_gather_channels', 'image_scatter_channels', 'image_gather_facets', 'image_scatter_facets']

import logging
from typing import List
import copy

import numpy
import xarray

from rascil.data_models.memory_data_models import Image

from rascil.processing_components.image.operations import create_image_from_array, create_empty_image_like, \
    image_is_canonical
from rascil.processing_components.image.iterators import image_raster_iter, image_channel_iter
from rascil.processing_components.image.image_selection import image_groupby, image_groupby_bins

log = logging.getLogger('rascil-logger')


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
    return [flat_facet for flat_facet in image_raster_iter(im, facets=facets, overlap=overlap,
                                                           taper=taper)]


def image_gather_facets(image_list: List[Image], im: Image, facets=1, overlap=0, taper=None,
                        return_flat=False):
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
        flats = [f for f in image_raster_iter(flat, facets=facets, overlap=overlap, taper=taper, make_flat=True)]
        
        sum_flats = create_empty_image_like(im)
        
        if return_flat:
            i = 0
            for sum_flat_facet in image_raster_iter(sum_flats, facets=facets, overlap=overlap, taper=taper):
                sum_flat_facet.data.values[...] += flats[i].data.values[...]
                i += 1
    
            return sum_flats
        else:
            i = 0
            for out_facet, sum_flat_facet in zip(image_raster_iter(out, facets=facets, overlap=overlap, taper=taper),
                                                 image_raster_iter(sum_flats, facets=facets, overlap=overlap,
                                                                   taper=taper)):
                out_facet.data.values[...] += flats[i].data.values * image_list[i].data.values[...]
                sum_flat_facet.data.values[...] += flats[i].data.values[...]
                i += 1
    
            out.data.values[sum_flats.data.values > 0.0] /= sum_flats.data.values[sum_flats.data.values > 0.0]
            out.data.values[sum_flats.data.values <= 0.0] = 0.0
        
            return out
    else:
        flat = create_empty_image_like(im)
        flat.data[...] = 1.0
        
        if return_flat:
            return flat
        else:
            for i, facet in enumerate(image_raster_iter(out, facets=facets, overlap=overlap, taper=taper)):
                facet.data[...].values += image_list[i].data[...].values
        
            return out


def image_scatter(im: Image, dim="frequency", subimages=None) -> List[Image]:
    """Scatter an image into a list of subimages using the dimension dim

    :param im: Image
    :param subimages: Number of channels
    :return: list of subimages

    See also
        :py:func:`rascil.processing_components.image.iterators.image_channel_iter`
    """
    image_list = image_groupby_bins(im, coordinate=dim, bins=subimages, squeeze=False)
    return list(image_list)


def image_scatter_channels(im: Image, subimages=None) -> List[Image]:
    """Scatter an image into a list of subimages using the channels

    :param im: Image
    :param subimages: Number of channels
    :return: list of subimages

    See also
        :py:func:`rascil.processing_components.image.iterators.image_channel_iter`
    """
    return image_scatter(im, "frequency")


def image_gather(image_list: List[Image], im: Image = None, subimages=0, dim="frequency") -> Image:
    """Gather a list of subimages back into an image using the channel_iterator
    
    If the template image is not given then it will be formed assuming that the list has
    been generated by image_scatter_channels with subimages = number of channels

    :param image_list: List of subimages
    :param im: Output image
    :param subimages: Number of image partitions on each axis (2)
    :return: list of subimages
    """
    image_data_list = [im.data for im in image_list]
    image_data_concat = xarray.concat(image_data_list, dim)
    if not isinstance(im, Image):
        im = copy.deepcopy(image_list[0])
    im.data = image_data_concat
    return im


def image_gather_channels(image_list: List[Image], im: Image = None, subimages=0) -> Image:
    """Gather a list of subimages back into an image using the channel_iterator

    If the template image is not given then it will be formed assuming that the list has
    been generated by image_scatter_channels with subimages = number of channels

    :param image_list: List of subimages
    :param im: Output image
    :param subimages: Number of image partitions on each axis (2)
    :return: list of subimages
    """
    for im in image_list:
        assert not numpy.isnan(numpy.sum(im.data.values)), "NaNs present in input images"

    image_data_list = [im.data for im in image_list]
    for ar in image_data_list:
        assert not numpy.isnan(numpy.sum(ar.values)), "NaNs present in input image data"
        
    if not isinstance(im, Image):
        im = copy.deepcopy(image_list[0])
    im.data = xarray.concat(image_data_list, "frequency")
    assert not numpy.isnan(numpy.sum(im.data.values)), "NaNs present in output image"
    return im

