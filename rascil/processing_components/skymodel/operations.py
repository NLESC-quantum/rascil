"""Function to manage skymodels.

"""

__all__ = [
    "copy_skymodel",
    "partition_skymodel_by_flux",
    "show_skymodel",
    "initialize_skymodel_voronoi",
    "calculate_skymodel_equivalent_image",
    "update_skymodel_from_gaintables",
    "update_skymodel_from_image",
    "expand_skymodel_by_skycomponents",
    "create_skymodel_from_skycomponents_gaintables",
    "extract_skycomponents_from_skymodel",
]

import logging
import warnings

import matplotlib.pyplot as plt
import numpy
from astropy.modeling import models, fitting
from astropy.wcs.utils import pixel_to_skycoord
from astropy.wcs.utils import skycoord_to_pixel

from rascil.data_models import get_parameter
from rascil.data_models.memory_data_models import SkyModel, Skycomponent, Image
from rascil.processing_components.calibration.operations import copy_gaintable
from rascil.processing_components.image.operations import smooth_image
from rascil.processing_components.skycomponent.base import copy_skycomponent
from rascil.processing_components.skycomponent.operations import (
    filter_skycomponents_by_flux,
    insert_skycomponent,
    image_voronoi_iter,
    find_skycomponents,
    fit_skycomponent
)

log = logging.getLogger("rascil-logger")


def copy_skymodel(sm):
    """Copy a sky model

    :param sm: SkyModel to be copied
    :return: SkyModel
    """
    if sm.components is not None:
        newcomps = [copy_skycomponent(comp) for comp in sm.components]
    else:
        newcomps = None
    
    if sm.image is not None:
        newimage = sm.image.copy(deep=True)
    else:
        newimage = None
    
    if sm.mask is not None:
        newmask = sm.mask.copy(deep=True)
    else:
        newmask = None
    
    if sm.gaintable is not None:
        newgt = copy_gaintable(sm.gaintable)
    else:
        newgt = None
    
    return SkyModel(
        components=newcomps,
        image=newimage,
        gaintable=newgt,
        mask=newmask,
        fixed=sm.fixed,
    )


def partition_skymodel_by_flux(sc, model, flux_threshold=-numpy.inf):
    """Partition skymodel according to flux

    Bright skycomponents are put into a SkyModel as a list, and weak skycomponents
    are inserted into SkyModel as an image.

    :param sc: List of skycomponents
    :param model: Model image
    :param flux_threshold:
    :return: SkyModel

    For example::

        fluxes = numpy.linspace(0, 1.0, 11)
        sc = [create_skycomponent(direction=phasecentre, flux=numpy.array([[f]]), frequency=frequency,
                                  polarisation_frame=PolarisationFrame('stokesI')) for f in fluxes]

        sm = partition_skymodel_by_flux(sc, model, flux_threshold=0.31)
        assert len(sm.components) == 7, len(sm.components)

    """
    brightsc = filter_skycomponents_by_flux(sc, flux_min=flux_threshold)
    weaksc = filter_skycomponents_by_flux(sc, flux_max=flux_threshold)
    log.info(
        "Converted %d components into %d bright components and one image containing %d components"
        % (len(sc), len(brightsc), len(weaksc))
    )
    im = model.copy(deep=True)
    im = insert_skycomponent(im, weaksc)
    return SkyModel(
        components=[copy_skycomponent(comp) for comp in brightsc],
        image=im.copy(deep=True),
        mask=None,
        fixed=False,
    )


def show_skymodel(sms, psf_width=1.75, cm="Greys", vmax=None, vmin=None):
    """Show a list of SkyModels

    :param sms: List of SkyModels
    :param psf_width: Width of PSF in pixels
    :param cm: matplotlib colormap
    :param vmax: Maximum in image display
    :param vmin: Minimum in image display
    :return:
    """
    sp = 1
    
    for ism, sm in enumerate(sms):
        plt.clf()
        plt.subplot(121, projection=sms[ism].image.image_acc.wcs.sub([1, 2]))
        sp += 1
        
        smodel = sms[ism].image.copy(deep=True)
        smodel = insert_skycomponent(smodel, sms[ism].components)
        smodel = smooth_image(smodel, psf_width)
        
        if vmax is None:
            vmax = numpy.max(smodel["pixels"].data[0, 0, ...])
        if vmin is None:
            vmin = numpy.min(smodel["pixels"].data[0, 0, ...])
        
        plt.imshow(
            smodel["pixels"].data[0, 0, ...],
            origin="lower",
            cmap=cm,
            vmax=vmax,
            vmin=vmin,
        )
        plt.xlabel(sms[ism].image.image_acc.wcs.wcs.ctype[0])
        plt.ylabel(sms[ism].image.image_acc.wcs.wcs.ctype[1])
        
        plt.title("SkyModel%d" % ism)
        
        components = sms[ism].components
        if components is not None:
            for sc in components:
                x, y = skycoord_to_pixel(
                    sc.direction, sms[ism].image.image_acc.wcs, 1, "wcs"
                )
                plt.plot(x, y, marker="+", color="red")
        
        gaintable = sms[ism].gaintable
        if gaintable is not None:
            plt.subplot(122)
            sp += 1
            phase = numpy.angle(sm.gaintable.gain[:, :, 0, 0, 0])
            phase -= phase[:, 0][:, numpy.newaxis]
            plt.imshow(phase, origin="lower")
            plt.xlabel("Dish/Station")
            plt.ylabel("Integration")
            plt.show()


def initialize_skymodel_voronoi(model, comps, gt=None):
    """Create a skymodel by Voronoi partitioning of the components, fill with components

    :param model: Model image
    :param comps: Skycomponents
    :param gt: Gaintable
    :return:

    For example::

        gaintable = create_gaintable_from_blockvisibility(block_vis)
        mpccal_skymodel = initialize_skymodel_voronoi(model, ical_components, gaintable)

    """
    skymodel_images = list()
    for i, mask in enumerate(image_voronoi_iter(model, comps)):
        im = model.copy(deep=True)
        im["pixels"].data *= mask["pixels"].data
        if gt is not None:
            newgt = copy_gaintable(gt)
            newgt.attrs["phasecentre"] = comps[i].direction
        else:
            newgt = None
        
        skymodel_images.append(
            SkyModel(image=im, components=None, gaintable=newgt, mask=mask)
        )
    
    return skymodel_images


def calculate_skymodel_equivalent_image(sm):
    """Calculate an equivalent image for a skymodel

    Uses the image from the first skymodel as the template for the image

    :param sm: List of skymodels
    :return: Image
    """
    combined_model = sm[0].image.copy(deep=True)
    combined_model["pixels"].data[...] = 0.0
    for th in sm:
        if th.image is not None:
            if th.mask is not None:
                combined_model["pixels"].data += (
                        th.mask["pixels"].data * th.image["pixels"].data
                )
            else:
                combined_model["pixels"].data += th.image["pixels"].data
    
    return combined_model


def update_skymodel_from_image(sm, im, damping=0.5):
    """Update a skymodel for an image, applying damping factor

    :param sm: List of skymodels
    :param im: Image
    :return: List of SkyModels
    """
    for i, th in enumerate(sm):
        newim = im.copy(deep=True)
        if th.mask is not None:
            newim["pixels"].data *= th.mask["pixels"].data
        th.image["pixels"].data += damping * newim["pixels"].data
    
    return sm


def update_skymodel_from_gaintables(sm, gt_list, calibration_context="T", damping=0.5):
    """Update a skymodel from a list of gaintables

    :param sm: List of skymodels
    :param gt_list: List of gain tables
    :param calibration_context: Type of gaintable e.g. 'T', 'G'
    :return: List of skymodels
    """
    assert len(sm) == len(gt_list)
    
    for i, th in enumerate(sm):
        # assert isinstance(th.gaintable, GainTable), th.gaintable
        th.gaintable["gain"].data *= numpy.exp(
            damping * 1j * numpy.angle(gt_list[i][calibration_context].gain)
        )
    
    return sm


def expand_skymodel_by_skycomponents(sm, **kwargs):
    """Expand a sky model so that all components and the image are in separate skymodels

    The mask and gaintable are taken to apply for all new skymodels.

    :param sm: SkyModel
    :return: List of SkyModels
    """
    
    def copy_image(im):
        """Copy an image

        :param im:
        :return:
        """
        if im is None:
            return None
        else:
            return im.copy(deep=True)
    
    result = [
        SkyModel(
            components=[comp],
            image=None,
            gaintable=copy_gaintable(sm.gaintable),
            mask=copy_image(sm.mask),
            fixed=sm.fixed,
        )
        for comp in sm.components
    ]
    if sm.image is not None:
        result.append(
            SkyModel(
                components=None,
                image=copy_image(sm.image),
                gaintable=copy_gaintable(sm.gaintable),
                mask=copy_image(sm.mask),
                fixed=sm.fixed,
            )
        )
    return result


def create_skymodel_from_skycomponents_gaintables(components, gaintables, **kwargs):
    """Create a list of sky model from lists of components and gaintables

    :param sm: SkyModel
    :return: List of SkyModels
    """
    assert len(components) == len(gaintables)
    result = [
        SkyModel(
            components=[copy_skycomponent(comp)],
            image=None,
            mask=None,
            gaintable=copy_gaintable(gaintables[icomp]),
        )
        for icomp, comp in enumerate(components)
    ]
    return result


def extract_skycomponents_from_skymodel(
        sm, im=None, **kwargs
):
    """Extract the bright components from the image in a skymodel

    This produces one component per frequency channel

    :param sm: skymodel
    :param im: image to be searched
    :param kwargs: Parameters for functions
    :param component_threshold: (in kwargs) Threshold in Jy to be classified as a source
    :param component_method: (in kwargs) Method to extract skycomponents: fit
    :return: Updated skymodel

    """
    component_threshold = get_parameter(kwargs, "component_threshold", None)
    if component_threshold is None:
        return sm
    
    component_method = get_parameter(kwargs, "component_method", None)
    if component_threshold > 1e12:
        component_threshold = None
        component_method = None
        
    if component_method is None:
        return sm
    elif component_method == "fit":
        log.info(
            f"extract_skycomponents_from_image: Searching and fitting for components > {component_threshold} Jy"
        )
    
        newsm = copy_skymodel(sm)
        try:
            if im is None:
                found_components = find_skycomponents(
                    newsm.image, fwhm=3, threshold=component_threshold
                )
                fitted_components = [fit_skycomponent(sm.image, sc, **kwargs) for sc in found_components]
            else:
                found_components = find_skycomponents(
                    im, fwhm=3, threshold=component_threshold
                )
                fitted_components = [fit_skycomponent(im, sc, **kwargs) for sc in found_components]

            number_points = len(fitted_components)
            if number_points > 0:
                log.info(
                    f"extract_skycomponents_from_image: Found {number_points} sources > {component_threshold} Jy"
                )
                newsm.components += fitted_components
        
        except AssertionError:
            log.warning(
                f"extract_skycomponents_from_image: No sources found > {component_threshold} Jy"
            )
            return sm
    else:
        raise ValueError(f"Unknown component extraction method {component_method}")
    
    return newsm


