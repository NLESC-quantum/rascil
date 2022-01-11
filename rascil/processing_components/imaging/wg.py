"""
Functions that implement prediction of and imaging from visibilities using the GPU-based gridder (WAGG version).

https://gitlab.com/ska-telescope/sdp/ska-gridder-nifty-cuda/-/tree/sim-874-python-wrapper

This performs all necessary w term corrections, to high precision.


"""

__all__ = ["predict_wg", "invert_wg"]

import copy
import logging
import sys

import numpy

from rascil.data_models.memory_data_models import BlockVisibility, Image
from rascil.data_models.parameters import get_parameter
from rascil.data_models.polarisation import convert_pol_frame
from rascil.processing_components.image.operations import image_is_canonical
from rascil.processing_components.imaging.base import (
    shift_vis_to_image,
    normalise_sumwt,
)
from rascil.processing_components.visibility.base import copy_visibility

log = logging.getLogger("rascil-logger")


def predict_wg(bvis: BlockVisibility, model: Image, **kwargs) -> BlockVisibility:
    """Predict using convolutional degridding.

    Nifty-gridder WAGG GPU version. https://gitlab.com/ska-telescope/sdp/ska-gridder-nifty-cuda

    In the imaging and pipeline workflows, this may be invoked using context='wg'.

    :param bvis: BlockVisibility to be predicted
    :param model: model image
    :return: resulting BlockVisibility (in place works)
    """

    try:
        import wagg as wg

        try:
            if not image_is_canonical(model):
                NonCanonicalError = ValueError
                raise NonCanonicalError
        except NonCanonicalError:
            log.error("The image model is not canonical. See the logfile. Exiting...")
            sys.exit("WG:The image model is not canonical")

        # If the model is None than just return the input blockvisibility.
        if model is None:
            return bvis

        nthreads = get_parameter(kwargs, "threads", 4)
        epsilon = get_parameter(kwargs, "epsilon", 1e-12)
        do_wstacking = get_parameter(kwargs, "do_wstacking", True)
        verbosity = get_parameter(kwargs, "verbosity", 0)

        newbvis = copy_visibility(bvis, zero=True)

        # Extracting data from BlockVisibility
        freq = bvis.frequency.data  # frequency, Hz
        nrows, nbaselines, vnchan, vnpol = bvis.vis.shape

        uvw = newbvis.uvw.data
        uvw = uvw.reshape([nrows * nbaselines, 3])
        uvw = numpy.nan_to_num(uvw)
        vis_temp = numpy.zeros([vnpol, vnchan, nbaselines * nrows], dtype="complex")

        # Get the image properties
        m_nchan, m_npol, ny, nx = model["pixels"].data.shape
        # Check if the number of frequency channels matches in bvis and a model
        try:
            if m_npol != vnpol:
                MatchError = ValueError
                raise MatchError
        except MatchError:
            log.error(
                "The number of frequency channels in bvis and a model does not match, exiting..."
            )
            sys.exit(
                "WG: The number of frequency channels in bvis and a model does not match"
            )

        flipped_uvw = copy.deepcopy(uvw)
        # We need to flip the u and w axes. The flip in w is equivalent to the conjugation of the
        # convolution function grid_visibility to griddata
        flipped_uvw[:, 0] *= -1.0
        flipped_uvw[:, 2] *= -1.0

        # Find out the image size/resolution
        pixsize = numpy.abs(numpy.radians(model.image_acc.wcs.wcs.cdelt[0]))

        # Make de-gridding over a frequency range and pol fields
        vis_to_im = numpy.round(
            model.image_acc.wcs.sub([4]).wcs_world2pix(freq, 0)[0]
        ).astype("int")

        if m_nchan == 1:
            for vpol in range(vnpol):
                vis_temp[vpol, :, :] = wg.dirty2ms(
                    flipped_uvw.astype(float),
                    bvis.frequency.data.astype(float),
                    model["pixels"].data[0, vpol, :, :].T.astype(float),
                    None,
                    pixsize,
                    pixsize,
                    epsilon,
                    do_wstacking,
                ).T
        else:
            for vpol in range(vnpol):
                for vchan in range(vnchan):
                    imchan = vis_to_im[vchan]
                    vis_temp[vpol, vchan, :] = wg.dirty2ms(
                        flipped_uvw.astype(float),
                        numpy.array(freq[vchan : vchan + 1]).astype(float),
                        model["pixels"].data[imchan, vpol, :, :].T.astype(float),
                        None,
                        pixsize,
                        pixsize,
                        epsilon,
                        do_wstacking,
                    )[:, 0]

        vis = convert_pol_frame(
            vis_temp.T,
            model.image_acc.polarisation_frame,
            bvis.blockvisibility_acc.polarisation_frame,
            polaxis=2,
        )

        vis = vis.reshape([nrows, nbaselines, vnchan, vnpol])
        newbvis["vis"].data = vis

        # Now we can shift the visibility from the image frame to the original visibility frame
        return shift_vis_to_image(newbvis, model, tangent=True, inverse=True)
    except ImportError:
        wagg = None
        print("WAGG module is not imported")
        return None


def invert_wg(
    bvis: BlockVisibility,
    model: Image,
    dopsf: bool = False,
    normalise: bool = True,
    **kwargs
) -> (Image, numpy.ndarray):
    """Invert using GPU-based WAGG nifty-gridder module

    Nifty-gridder WAGG GPU version. https://gitlab.com/ska-telescope/sdp/ska-gridder-nifty-cuda

    Use the image im as a template. Do PSF in a separate call.

    In the imaging and pipeline workflows, this may be invoked using context='wg'.

    :param dopsf: Make the PSF instead of the dirty image
    :param bvis: BlockVisibility to be inverted
    :param im: image template (not changed)
    :param normalise: normalise by the sum of weights (True)
    :return: (resulting image, sum of the weights for each frequency and polarization)

    """
    try:
        import wagg as wg

        try:
            if not image_is_canonical(model):
                NonCanonicalError = ValueError
                raise NonCanonicalError
        except NonCanonicalError:
            log.error("The image model is not canonical. See the logfile. Exiting...")
            sys.exit("WG:The image model is not canonical")

        im = model.copy(deep=True)

        nthreads = get_parameter(kwargs, "threads", 4)
        epsilon = get_parameter(kwargs, "epsilon", 1e-12)
        do_wstacking = get_parameter(kwargs, "do_wstacking", True)
        verbosity = get_parameter(kwargs, "verbosity", 0)

        bvis_shifted = copy_visibility(bvis)
        bvis_shifted = shift_vis_to_image(bvis_shifted, im, tangent=True, inverse=False)

        freq = bvis_shifted.frequency.data  # frequency, Hz

        nrows, nbaselines, vnchan, vnpol = bvis_shifted.vis.shape

        ms = bvis_shifted.blockvisibility_acc.flagged_vis
        ms = ms.reshape([nrows * nbaselines, vnchan, vnpol])
        ms = convert_pol_frame(
            ms,
            bvis.blockvisibility_acc.polarisation_frame,
            im.image_acc.polarisation_frame,
            polaxis=2,
        ).astype("c16")

        uvw = copy.deepcopy(bvis_shifted.uvw.data)
        uvw = uvw.reshape([nrows * nbaselines, 3])

        weight = bvis_shifted.blockvisibility_acc.flagged_imaging_weight.astype("f8")
        weight = weight.reshape([nrows * nbaselines, vnchan, vnpol])

        # Find out the image size/resolution
        npixdirty = im["pixels"].data.shape[-1]
        pixsize = numpy.abs(numpy.radians(im.image_acc.wcs.wcs.cdelt[0]))

        flipped_uvw = copy.deepcopy(uvw)
        # We need to flip the u and w axes.
        flipped_uvw[:, 0] *= -1.0
        flipped_uvw[:, 2] *= -1.0

        nchan, npol, ny, nx = im["pixels"].data.shape
        im["pixels"].data[...] = 0.0
        sum_weight = numpy.zeros([nchan, npol])

        # Set up the conversion from visibility channels to image channels
        vis_to_im = numpy.round(
            model.image_acc.wcs.sub([4]).wcs_world2pix(freq, 0)[0]
        ).astype("int")

        mfs = nchan == 1 and vnchan > 1
        ms_temp = ms.T
        weight_temp = weight.T
        if dopsf:
            ms_temp[...] = 0.0
            ms_temp[0, ...] = 1.0

        # lms and lwt - the contiguous versions of ms_temp and weight_temp
        if mfs:
            for pol in range(npol):
                lms = numpy.ascontiguousarray(ms_temp[pol, :, :].T)
                if numpy.max(numpy.abs(lms)) > 0.0:
                    lwt = numpy.ascontiguousarray(weight_temp[pol, :, :].T)
                    dirty = wg.ms2dirty(
                        flipped_uvw,
                        bvis.frequency.data,
                        lms,
                        lwt,
                        npixdirty,
                        npixdirty,
                        pixsize,
                        pixsize,
                        epsilon,
                        do_wstacking,
                    )
                    im["pixels"].data[0, pol] += dirty.T
                sum_weight[0, pol] += numpy.sum(weight_temp[pol, :, :].T)
        else:
            for pol in range(npol):
                for vchan in range(vnchan):
                    ichan = vis_to_im[vchan]
                    frequency = numpy.array(freq[vchan : vchan + 1]).astype(float)
                    lms = numpy.ascontiguousarray(ms_temp[pol, vchan, :, numpy.newaxis])
                    if numpy.max(numpy.abs(lms)) > 0.0:
                        lwt = numpy.ascontiguousarray(
                            weight_temp[pol, vchan, :, numpy.newaxis]
                        )
                        dirty = wg.ms2dirty(
                            flipped_uvw,
                            frequency,
                            lms,
                            lwt,
                            npixdirty,
                            npixdirty,
                            pixsize,
                            pixsize,
                            epsilon,
                            do_wstacking,
                        )
                        im["pixels"].data[ichan, pol] += dirty.T
                    sum_weight[ichan, pol] += numpy.sum(
                        weight_temp[pol, vchan, :].T, axis=0
                    )

        if normalise:
            im = normalise_sumwt(im, sum_weight)

        return im, sum_weight

    except ImportError:
        wagg = None
        print("WAGG module is not imported")
        return None
