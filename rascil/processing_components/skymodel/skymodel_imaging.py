import logging

import numpy

from rascil.data_models import get_parameter
from rascil.processing_components import (
    copy_skycomponent,
    apply_beam_to_skycomponent,
    normalise_sumwt,
    extract_skycomponents_from_skymodel,
    image_scatter_facets,
    image_gather_facets,
)
from rascil.processing_components.skycomponent.taylor_terms import (
    find_skycomponents_frequency_taylor_terms,
)
from rascil.processing_components.imaging.imaging import (
    predict_blockvisibility,
    invert_blockvisibility,
)
from rascil.processing_components.calibration import apply_gaintable
from rascil.processing_components.image import restore_cube, fit_psf
from rascil.processing_components.imaging import dft_skycomponent_visibility
from rascil.processing_components.skycomponent.operations import restore_skycomponent
from rascil.processing_components.visibility import (
    copy_visibility,
    concatenate_visibility,
)


def skymodel_predict_calibrate(
    bvis, skymodel, context="ng", docal=False, inverse=True, get_pb=None, **kwargs
):
    """Predict visibility for a skymodel, optionally apply calibration

    A skymodel consists of an image and a list of components, each optionally with
    its own gaintable.

    The function get_pb should have the signature:

        get_pb(BlockVisibility, Image)

    and should return the primary beam for the blockvisibility.

    :param bvis: Input visibility
    :param skymodel: Skymodel
    :param context: Imaging context 2d or ng or awprojection
    :param get_pb: Function to get a primary beam
    :param docal: Apply calibration table in skymodel
    :param kwargs: Parameters for functions in components
    :return: Visibility with dft of components, fft of image, gaintable applied (optional)
    """
    v = copy_visibility(bvis, zero=True)

    vis_slices = []
    if get_pb is not None:
        for time, vis_slice in v.groupby("time", squeeze=False):

            pb = get_pb(vis_slice, skymodel.image)

            # First do the DFT for the components
            if len(skymodel.components) > 0:
                if skymodel.mask is not None or pb is not None:
                    comps = copy_skycomponent(skymodel.components)
                    if skymodel.mask is not None:
                        comps = apply_beam_to_skycomponent(comps, skymodel.mask)
                    if pb is not None:
                        comps = apply_beam_to_skycomponent(comps, pb)
                    vis_slice = dft_skycomponent_visibility(vis_slice, comps, **kwargs)
                else:
                    vis_slice = dft_skycomponent_visibility(
                        vis_slice, skymodel.components, **kwargs
                    )

            # Now do the FFT of the image, after multiplying by the mask and primary
            # beam
            if skymodel.image is not None:
                if numpy.max(numpy.abs(skymodel.image["pixels"].data)) > 0.0:
                    imgv = copy_visibility(vis_slice, zero=True)
                    if skymodel.mask is not None or pb is not None:
                        model = skymodel.image.copy(deep=True)
                        if skymodel.mask is not None:
                            model["pixels"].data *= skymodel.mask["pixels"].data
                        if pb is not None:
                            model["pixels"].data *= pb["pixels"].data
                        imgv = predict_blockvisibility(
                            imgv, model, context=context, **kwargs
                        )
                    else:
                        imgv = predict_blockvisibility(
                            imgv, skymodel.image, context=context, **kwargs
                        )
                    vis_slice["vis"].data += imgv["vis"].data

            vis_slices.append(vis_slice)

        v = concatenate_visibility(vis_slices, "time")

        if docal and skymodel.gaintable is not None:
            v = apply_gaintable(v, skymodel.gaintable, inverse=inverse)

        return v

    else:

        # First do the DFT or the components
        if len(skymodel.components) > 0:
            if skymodel.mask is not None:
                comps = copy_skycomponent(skymodel.components)
                comps = apply_beam_to_skycomponent(comps, skymodel.mask)
                v = dft_skycomponent_visibility(v, comps, **kwargs)
            else:
                v = dft_skycomponent_visibility(v, skymodel.components, **kwargs)

        # Now do the FFT of the image, after multiplying by the mask and primary
        # beam
        if skymodel.image is not None:
            if numpy.max(numpy.abs(skymodel.image["pixels"].data)) > 0.0:
                imgv = copy_visibility(v, zero=True)
                if skymodel.mask is not None:
                    model = skymodel.image.copy(deep=True)
                    model["pixels"].data *= skymodel.mask["pixels"].data
                    imgv = predict_blockvisibility(
                        imgv, model, context=context, **kwargs
                    )
                else:
                    imgv = predict_blockvisibility(
                        imgv, skymodel.image, context=context, **kwargs
                    )
                v["vis"].data += imgv["vis"].data

        if docal and skymodel.gaintable is not None:
            v = apply_gaintable(v, skymodel.gaintable, inverse=inverse)

        return v


def skymodel_calibrate_invert(
    bvis,
    skymodel,
    context="ng",
    docal=False,
    get_pb=None,
    normalise=True,
    flat_sky=False,
    **kwargs
):
    """Inverse Fourier sum of visibility to image and components

    :param bvis: Visibility to be transformed
    :param skymodel: Skymodel
    :return: Skymodel containing transforms
    """

    if docal and skymodel.gaintable is not None:
        bvis = apply_gaintable(bvis, skymodel.gaintable)

    if skymodel.image is None:
        raise ValueError("skymodel image is None")

    sum_flats = skymodel.image.copy(deep=True)
    sum_flats["pixels"][...] = 0.0
    sum_dirtys = skymodel.image.copy(deep=True)
    sum_dirtys["pixels"][...] = 0.0

    if get_pb is not None:
        for time, vis_slice in bvis.groupby("time", squeeze=False):

            pb = get_pb(vis_slice, skymodel.image)

            # Just do a straightforward invert for just this blockvis
            # and then apply the mask and primary beam if present
            # The return value result contains the weighted image and
            # the weights as an image (including mask and primary beam)
            result = invert_blockvisibility(
                vis_slice, skymodel.image, context=context, normalise=False, **kwargs
            )
            flat = numpy.ones_like(result[0]["pixels"].data)
            if skymodel.mask is not None:
                flat *= skymodel.mask["pixels"].data
            if pb is not None:
                flat *= pb["pixels"].data

            # We need to apply the flat to the dirty image
            sum_dirtys["pixels"].data += flat * result[0]["pixels"].data
            # The sum_flats should contain the weights and the square of the PB
            sum_flats["pixels"].data += (
                flat * flat * result[1][:, :, numpy.newaxis, numpy.newaxis]
            )
        if normalise:
            sum_dirtys = normalise_sumwt(sum_dirtys, sum_flats, flat_sky=flat_sky)
            sum_flats["pixels"].data = numpy.sqrt(sum_flats["pixels"].data)

        return (sum_dirtys, sum_flats)

    else:
        result = invert_blockvisibility(bvis, skymodel.image, context=context, **kwargs)
        if skymodel.mask is not None:
            result[0]["pixels"].data *= skymodel.mask["pixels"].data

        return result
