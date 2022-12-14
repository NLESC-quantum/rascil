__all__ = [
    "image_rsexecute_map_workflow",
    "sum_images_rsexecute",
    "image_gather_channels_rsexecute",
]

import logging

from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute
from rascil.processing_components.image import (
    image_scatter_facets,
    image_gather_facets,
    image_gather_channels,
)

log = logging.getLogger("rascil-logger")


def image_rsexecute_map_workflow(
    im, imfunction, facets=1, overlap=0, taper=None, **kwargs
):
    """Apply a function across an image: scattering to subimages, applying the function, and then gathering

    :param im: Image to be processed
    :param imfunction: Function to be applied
    :param facets: See image_scatter_facets
    :param overlap: image_scatter_facets
    :param taper: image_scatter_facets
    :param kwargs: kwargs for imfunction
    :return: graph for output image

    For example::

        rsexecute.set_client(use_dask=True)
        model = create_test_image(frequency=frequency, phasecentre=phasecentre, cellsize=0.001,
                                         polarisation_frame=PolarisationFrame('stokesI'))
        def imagerooter(im, **kwargs):
            im["pixels"].data = numpy.sqrt(numpy.abs(im["pixels"].data))
            return im
        root_graph = image_rsexecute_map_workflow(model, imagerooter, facets=16)
        root_image = rsexecute.compute(root_graph, sync=True)

    """

    facets_list = rsexecute.execute(image_scatter_facets, nout=facets**2)(
        im, facets=facets, overlap=overlap, taper=taper
    )
    root_list = [
        rsexecute.execute(imfunction)(facet, **kwargs) for facet in facets_list
    ]
    gathered = rsexecute.execute(image_gather_facets)(
        root_list, im, facets=facets, overlap=overlap, taper=taper
    )
    return gathered


def sum_images_rsexecute(image_list, split=2):
    """Sum a set of images, using a tree reduction

    :param image_list: List of images
    :return: graph for summed image

    """

    def sum_images(imlist):
        if len(imlist) == 1:
            return imlist[0]
        else:
            assert len(imlist) > 1, imlist
            out = imlist[0].copy(deep=True)
            out["pixels"].data += imlist[1]["pixels"].data
            return out

    if len(image_list) > split:
        centre = len(image_list) // split
        result = [
            sum_images_rsexecute(image_list[:centre]),
            sum_images_rsexecute(image_list[centre:]),
        ]
        return rsexecute.execute(sum_images, nout=2)(result)
    else:
        return rsexecute.execute(sum_images, nout=2)(image_list)


def image_gather_channels_rsexecute(image_list, split=0):
    """Gather a set of images in frequency, using a tree reduction or directly

    :param image_list: List of images
    :param split: Order of split i.e. 2 is binary, 0 is list
    :return: graph for summed image
    """

    if split == 0:
        return rsexecute.execute(image_gather_channels)(image_list)

    else:

        def concat_channel_images(image_list):
            if len(image_list) == 1:
                return image_list[0]
            else:
                assert len(image_list) > 1, image_list
                return image_gather_channels(image_list)

        if len(image_list) > split:
            centre = len(image_list) // split
            result = [
                image_gather_channels_rsexecute(image_list[:centre], split=split),
                image_gather_channels_rsexecute(image_list[centre:], split=split),
            ]
            return rsexecute.execute(concat_channel_images, nout=2)(result)
        else:
            return rsexecute.execute(concat_channel_images, nout=2)(image_list)
