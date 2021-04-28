""" Workflows needed for skymodel mpc functions

"""

import numpy

from rascil.processing_components import (
    copy_visibility,
    copy_skycomponent,
    apply_beam_to_skycomponent,
    dft_skycomponent_visibility,
)
from rascil.workflows.serial import predict_list_serial_workflow, invert_list_serial_workflow
from rascil.workflows.rsexecute.execution_support import rsexecute


def crosssubtract_datamodels_skymodel_list_rsexecute_workflow(obsvis, modelvis_list):
    """Form data models by subtracting sum from the observed and adding back each model in turn

    vmodel[p] = vobs - sum(i!=p) modelvis[i]

    This is the E step in the Expectation-Maximisation algorithm.

    :param obsvis: "Observed" visibility
    :param modelvis_list: List of BlockVisibility data model predictions
    :return: List of (image, weight) tuples)
    """

    # Now do the meaty part. We probably want to refactor this for performance once it works.
    def skymodel_mpc_vsum(ov, mv):
        # Observed vis minus the sum of all predictions
        verr = copy_visibility(ov)
        for m in mv:
            verr["vis"].data -= m["vis"].data
        # Now add back each model in turn
        result = list()
        for m in mv:
            vr = copy_visibility(verr)
            vr["vis"].data += m["vis"].data
            result.append(vr)
        assert len(result) == len(mv)
        return result

    return rsexecute.execute(skymodel_mpc_vsum, nout=len(modelvis_list))(obsvis, modelvis_list)


def convolve_skymodel_list_rsexecute_workflow(
    obsvis, skymodel_list, context="ng", gcfcf=None, **kwargs
):
    """Form residual image from observed visibility and a set of skymodel without calibration

    This is similar to convolving the skymodel images with the PSF

    :param vis_list: List of BlockVisibility data models
    :param skymodel_list: skymodel list
    :param context: Type of processing e.g. 2d, wstack, timeslice or facets
    :param gcfcg: tuple containing grid correction and convolution function
    :param docal: Apply calibration table in skymodel
    :param kwargs: Parameters for functions in components
    :return: List of (image, weight) tuples)
    """

    def skymodel_predict_invert(ov, sm, g):
        # assert isinstance(ov, BlockVisibility), ov
        # assert isinstance(sm, SkyModel), sm
        if g is not None:
            assert len(g) == 2, g
            # assert isinstance(g[0], Image), g[0]
            # assert isinstance(g[1], ConvolutionFunction), g[1]

        v = copy_visibility(ov)

        v["vis"].data[...] = 0.0 + 0.0j

        if len(sm.components) > 0:

            if sm.mask is not None:
                comps = copy_skycomponent(sm.components)
                comps = apply_beam_to_skycomponent(comps, sm.mask)
                v = dft_skycomponent_visibility(v, comps)
            else:
                v = dft_skycomponent_visibility(v, sm.components)

        if sm.image is not None:
            if numpy.max(numpy.abs(sm.image["pixels"].data)) > 0.0:
                if sm.mask is not None:
                    model = sm.image.copy(deep=True)
                    model["pixels"].data *= sm.mask["pixels"].data
                else:
                    model = sm.image
                v = predict_list_serial_workflow(
                    [v], [model], context=context, gcfcf=[g], **kwargs
                )[0]

        result = invert_list_serial_workflow(
            [v], [sm.image], context=context, gcfcf=[g], **kwargs
        )[0]
        if sm.mask is not None:
            result[0]["pixels"].data *= sm.mask["pixels"].data
        return result

    if gcfcf is None:
        return [
            rsexecute.execute(skymodel_predict_invert, nout=len(skymodel_list))(obsvis, sm, None)
            for ism, sm in enumerate(skymodel_list)
        ]
    else:
        return [
            rsexecute.execute(skymodel_predict_invert, nout=len(skymodel_list))(
                obsvis, sm, gcfcf[ism]
            )
            for ism, sm in enumerate(skymodel_list)
        ]
