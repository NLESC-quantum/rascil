""" BlockVisibility operations

"""

__all__ = [
    "concatenate_blockvisibility_frequency",
    "concatenate_visibility",
    "subtract_visibility",
    "qa_visibility",
    "remove_continuum_blockvisibility",
    "divide_visibility",
    "integrate_visibility_by_channel",
    "average_blockvisibility_by_channel",
    "convert_blockvisibility_to_stokes",
    "convert_blockvisibility_to_stokes",
    "convert_blockvisibility_to_stokesI",
    "convert_blockvisibility_stokesI_to_polframe",
]

import copy
import collections
import logging
from typing import List

import numpy
import xarray

from rascil.data_models.memory_data_models import BlockVisibility, QA
from rascil.data_models.polarisation import (
    convert_linear_to_stokes,
    convert_circular_to_stokesI,
    convert_linear_to_stokesI,
    convert_circular_to_stokes,
    PolarisationFrame,
)
from rascil.processing_components.visibility import copy_visibility

log = logging.getLogger("rascil-logger")


def concatenate_visibility(vis_list, dim="time"):
    """Concatenate a list of visibilities

    :param vis_list: List of vis
    :return: Concatendated visibility
    """
    if not len(vis_list) > 0:
        raise ValueError("concatenate_visibility: vis_list is empty")

    try:
        return xarray.concat(
            vis_list, dim=dim, data_vars="minimal", coords="minimal", compat="override"
        )
    except TypeError:
        # RASCIL-defined classes that inherit from xarray.Dataset, do not
        # take attrs and an input argument; xarray.concat tries to call the
        # subclass with attrs arg, because it assumes that the subclass takes
        # the same args as the Dataset class; we need to manually accommodate for this
        return xarray.concat(
            [dataset.to_native_dataset() for dataset in vis_list],
            dim=dim,
            data_vars="minimal",
            coords="minimal",
            compat="override",
        )


def concatenate_blockvisibility_frequency(bvis_list):
    """Concatenate a list of BlockVisibility's in frequency

    The list should be in sequence of channels

    :param bvis_list: List of BlockVisibility
    :return: BlockVisibility
    """
    return concatenate_visibility(bvis_list, "frequency")


def subtract_visibility(vis, model_vis, inplace=False):
    """Subtract model_vis from vis, returning new visibility

    :param vis:
    :param model_vis:
    :return:
    """
    # assert isinstance(vis, BlockVisibility), vis
    # assert isinstance(model_vis, BlockVisibility), model_vis

    assert (
        vis.vis.shape == model_vis.vis.shape
    ), "Observed %s and model visibilities %s have different shapes" % (
        vis.vis.shape,
        model_vis.vis.shape,
    )

    if inplace:
        vis["vis"].data = vis["vis"].data - model_vis["vis"].data
        return vis
    else:
        residual_vis = copy_visibility(vis)
        residual_vis["vis"].data = residual_vis["vis"].data - model_vis["vis"].data
        return residual_vis


def qa_visibility(vis: BlockVisibility, context=None) -> QA:
    """Assess the quality of Visibility

    :param context:
    :param vis: blockvisibility to be assessed
    :return: QA
    """
    # assert isinstance(vis, BlockVisibility), vis

    avis = numpy.abs(vis["vis"].data)
    data = {
        "maxabs": numpy.max(avis),
        "minabs": numpy.min(avis),
        "rms": numpy.std(avis),
        "medianabs": numpy.median(avis),
    }
    qa = QA(origin="qa_visibility", data=data, context=context)
    return qa


def remove_continuum_blockvisibility(
    vis: BlockVisibility, degree=1, mask=None
) -> BlockVisibility:
    """Fit and remove continuum visibility

    Fit a polynomial in frequency of the specified degree where mask is True

    :param vis: BlockVisibility
    :param degree: Degree of polynomial
    :param mask: Mask of continuum
    :return: BlockVisibility
    """
    # assert isinstance(vis, BlockVisibility), vis

    if mask is not None:
        assert numpy.sum(mask) > 2 * degree, "Insufficient channels for fit"

    nchan = len(vis.frequency)
    # TODO: optimise loop
    x = (vis.frequency - vis.frequency[nchan // 2]) / (
        vis.frequency[0] - vis.frequency[nchan // 2]
    )
    for row in range(vis.nvis):
        for ibaseline, baseline in enumerate(vis.baselines):
            for pol in range(vis.blockvisibility_acc.polarisation_frame.npol):
                wt = numpy.sqrt(
                    vis.blockvisibility_acc.flagged_weight[row, ibaseline, :, pol]
                )
                if mask is not None:
                    wt[mask] = 0.0
                fit = numpy.polyfit(
                    x, vis["vis"][row, ibaseline, :, pol], w=wt, deg=degree
                )
                prediction = numpy.polyval(fit, x)
                vis["vis"][row, ibaseline, :, pol] -= prediction
    return vis


def divide_visibility(vis: BlockVisibility, modelvis: BlockVisibility):
    """Divide visibility by model forming visibility for equivalent point source

    This is a useful intermediate product for calibration. Variation of the visibility in time and
    frequency due to the model structure is removed and the data can be averaged to a limit determined
    by the instrumental stability. The weight is adjusted to compensate for the division.

    Zero divisions are avoided and the corresponding weight set to zero.

    :param vis:
    :param modelvis:
    :return:
    """
    # assert isinstance(vis, BlockVisibility), vis

    x = numpy.zeros_like(vis.blockvisibility_acc.flagged_vis)
    xwt = (
        numpy.abs(modelvis.blockvisibility_acc.flagged_vis) ** 2
        * vis.blockvisibility_acc.flagged_weight
    )
    mask = xwt > 0.0
    x[mask] = (
        vis.blockvisibility_acc.flagged_vis[mask]
        / modelvis.blockvisibility_acc.flagged_vis[mask]
    )

    pointsource_vis = BlockVisibility(
        flags=vis.flags.data,
        baselines=vis.baselines,
        frequency=vis.frequency.data,
        channel_bandwidth=vis.channel_bandwidth.data,
        phasecentre=vis.phasecentre,
        configuration=vis.configuration,
        uvw=vis.uvw.data,
        time=vis.time.data,
        integration_time=vis.integration_time.data,
        vis=x,
        weight=xwt,
        imaging_weight=vis.imaging_weight,
        source=vis.source,
        meta=vis.meta,
        polarisation_frame=vis.blockvisibility_acc.polarisation_frame,
    )
    return pointsource_vis


def integrate_visibility_by_channel(vis: BlockVisibility) -> BlockVisibility:
    """Integrate visibility across all channels, returning new visibility

    :param vis: BlockVisibility
    :return: BlockVisibility
    """

    # assert isinstance(vis, BlockVisibility), vis

    vis_shape = list(vis.vis.shape)
    ntimes, nbaselines, nchan, npol = vis_shape
    vis_shape[-2] = 1
    # newvis['flags'].data[..., 0, :] = numpy.sum(vis.flags.data, axis=-2)
    # newvis['flags'].data[newvis['flags'].data < nchan] = 0
    # newvis['flags'].data[newvis['flags'].data > 1] = 1
    flags = numpy.sum(vis.flags.data, axis=-2)[..., numpy.newaxis, :]
    flags[flags < nchan] = 0
    flags[flags > 1] = 1

    newvis = numpy.sum(
        vis["vis"].data * vis.blockvisibility_acc.flagged_weight, axis=-2
    )[..., numpy.newaxis, :]
    newweights = numpy.sum(vis.blockvisibility_acc.flagged_weight, axis=-2)[
        ..., numpy.newaxis, :
    ]
    newimaging_weights = numpy.sum(
        vis.blockvisibility_acc.flagged_imaging_weight, axis=-2
    )[..., numpy.newaxis, :]
    mask = (1 - flags) * newweights > 0.0
    newvis[mask] = newvis[mask] / ((1 - flags) * newweights)[mask]

    return BlockVisibility(
        frequency=numpy.ones([1]) * numpy.average(vis.frequency.data),
        channel_bandwidth=numpy.ones([1]) * numpy.sum(vis.channel_bandwidth.data),
        baselines=vis.baselines,
        phasecentre=vis.phasecentre,
        configuration=vis.configuration,
        uvw=vis.uvw.data,
        time=vis.time.data,
        vis=newvis,
        flags=flags,
        weight=newweights,
        imaging_weight=newimaging_weights,
        integration_time=vis.integration_time.data,
        polarisation_frame=vis.blockvisibility_acc.polarisation_frame,
        source=vis.source,
        meta=vis.meta,
    )


def average_blockvisibility_by_channel(
    vis: BlockVisibility, channel_average=None
) -> List[BlockVisibility]:
    """Average visibility by groups of channels, returning list of new visibility

    :param vis: BlockVisibility
    :param channel_average: Number of channels to average
    :return: List[BlockVisibility]
    """

    # assert isinstance(vis, BlockVisibility), vis

    vis_shape = list(vis.vis.shape)
    ntimes, nbaselines, nchan, npol = vis_shape

    newvis_list = list()
    ochannels = range(nchan)

    channels = []
    for i in range(0, nchan, channel_average):
        channels.append([ochannels[i], ochannels[i + channel_average - 1] + 1])
    for group in channels:
        vis_shape[-2] = 1
        freq = numpy.array([numpy.average(vis.frequency[group[0] : group[1]])])
        cb = numpy.array([numpy.sum(vis.channel_bandwidth[group[0] : group[1]])])
        newvis = BlockVisibility(
            frequency=freq,
            channel_bandwidth=cb,
            baselines=vis.baselines,
            phasecentre=vis.phasecentre,
            configuration=vis.configuration,
            uvw=vis.uvw,
            time=vis.time,
            vis=numpy.zeros(vis_shape, dtype="complex"),
            flags=numpy.zeros(vis_shape, dtype="int"),
            weight=numpy.zeros(vis_shape, dtype="float"),
            imaging_weight=numpy.zeros(vis_shape, dtype="float"),
            integration_time=vis.integration_time,
            polarisation_frame=vis.blockvisibility_acc.polarisation_frame,
            source=vis.source,
            meta=vis.meta,
        )
        vf = vis.flags[..., group[0] : group[1], :]
        vfvw = (
            vis.blockvisibility_acc.flagged_vis[..., group[0] : group[1], :]
            * vis.weight[..., group[0] : group[1], :]
        )
        vfw = vis.blockvisibility_acc.flagged_weight[..., group[0] : group[1], :]
        vfiw = vis.blockvisibility_acc.flagged_imaging_weight[
            ..., group[0] : group[1], :
        ]

        newvis["flags"][..., 0, :] = numpy.sum(vf, axis=-2)
        newvis["flags"][newvis["flags"] < nchan] = 0
        newvis["flags"][newvis["flags"] > 1] = 1

        newvis["vis"][..., 0, :] = numpy.sum(vfvw, axis=-2)
        newvis["weight"][..., 0, :] = numpy.sum(vfw, axis=-2)
        newvis["imaging_weight"][..., 0, :] = numpy.sum(vfiw, axis=-2)
        mask = newvis.blockvisibility_acc.flagged_weight > 0.0
        newvis["vis"][mask] = (
            newvis["vis"][mask] / newvis.blockvisibility_acc.flagged_weight[mask]
        )

        newvis_list.append(newvis)

    return newvis_list


def convert_blockvisibility_to_stokes(vis):
    """Convert the polarisation frame data into Stokes parameters.

    :param vis: blockvisibility
    :return: Converted visibility data.
    """
    poldef = vis.blockvisibility_acc.polarisation_frame
    if poldef == PolarisationFrame("linear"):
        vis["vis"].data[...] = convert_linear_to_stokes(vis["vis"].data, polaxis=3)
        vis["flags"].data[...] = numpy.logical_or(
            vis.flags.data[..., 0], vis.flags.data[..., 3]
        )[..., numpy.newaxis]
        vis.attrs["polarisation_frame"] = PolarisationFrame("stokesIQUV")
    elif poldef == PolarisationFrame("circular"):
        vis["vis"].data[...] = convert_circular_to_stokes(vis["vis"].data, polaxis=3)
        vis["flags"].data[...] = numpy.logical_or(
            vis.flags.data[..., 0], vis.flags.data[..., 3]
        )[..., numpy.newaxis]
        vis.attrs["polarisation_frame"] = PolarisationFrame("stokesIQUV")
    return vis


def convert_blockvisibility_to_stokesI(vis):
    """Convert the polarisation frame data into Stokes I dropping other polarisations, return new Visibility

    :param vis: blockvisibility
    :return: Converted visibility data.
    """
    if vis.blockvisibility_acc.polarisation_frame == PolarisationFrame("stokesI"):
        return vis

    polarisation_frame = PolarisationFrame("stokesI")
    poldef = vis.blockvisibility_acc.polarisation_frame
    if poldef == PolarisationFrame("linear"):
        vis_data = convert_linear_to_stokesI(vis.blockvisibility_acc.flagged_vis)
        vis_flags = numpy.logical_or(vis.flags.data[..., 0], vis.flags.data[..., 3])[
            ..., numpy.newaxis
        ]
        vis_weight = (
            vis.blockvisibility_acc.flagged_weight[..., 0]
            + vis.blockvisibility_acc.flagged_weight[..., 3]
        )[..., numpy.newaxis]
        vis_imaging_weight = (
            vis.blockvisibility_acc.flagged_imaging_weight[..., 0]
            + vis.blockvisibility_acc.flagged_imaging_weight[..., 3]
        )[..., numpy.newaxis]
    elif poldef == PolarisationFrame("linearnp"):
        vis_data = convert_linear_to_stokesI(vis.blockvisibility_acc.flagged_vis)
        vis_flags = numpy.logical_or(vis.flags.data[..., 0], vis.flags.data[..., 1])[
            ..., numpy.newaxis
        ]
        vis_weight = (
            vis.blockvisibility_acc.flagged_weight[..., 0]
            + vis.blockvisibility_acc.flagged_weight[..., 1]
        )[..., numpy.newaxis]
        vis_imaging_weight = (
            vis.blockvisibility_acc.flagged_imaging_weight[..., 0]
            + vis.blockvisibility_acc.flagged_imaging_weight[..., 1]
        )[..., numpy.newaxis]
    elif poldef == PolarisationFrame("circular"):
        vis_data = convert_circular_to_stokesI(vis.blockvisibility_acc.flagged_vis)
        vis_flags = numpy.logical_or(vis.flags.data[..., 0], vis.flags.data[..., 3])[
            ..., numpy.newaxis
        ]
        vis_weight = (
            vis.blockvisibility_acc.flagged_weight[..., 0]
            + vis.blockvisibility_acc.flagged_weight[..., 3]
        )[..., numpy.newaxis]
        vis_imaging_weight = (
            vis.blockvisibility_acc.flagged_imaging_weight[..., 0]
            + vis.blockvisibility_acc.flagged_imaging_weight[..., 3]
        )[..., numpy.newaxis]
    elif poldef == PolarisationFrame("circularnp"):
        vis_data = convert_circular_to_stokesI(vis.blockvisibility_acc.flagged_vis)
        vis_flags = numpy.logical_or(vis.flags.data[..., 0], vis.flags.data[..., 1])[
            ..., numpy.newaxis
        ]
        vis_weight = (
            vis.blockvisibility_acc.flagged_weight[..., 0]
            + vis.blockvisibility_acc.flagged_weight[..., 1]
        )[..., numpy.newaxis]
        vis_imaging_weight = (
            vis.blockvisibility_acc.flagged_imaging_weight[..., 0]
            + vis.blockvisibility_acc.flagged_imaging_weight[..., 1]
        )[..., numpy.newaxis]
    else:
        raise NameError("Polarisation frame %s unknown" % poldef)

    return BlockVisibility(
        frequency=vis.frequency.data,
        channel_bandwidth=vis.channel_bandwidth.data,
        phasecentre=vis.phasecentre,
        baselines=vis["baselines"],
        configuration=vis.attrs["configuration"],
        uvw=vis["uvw"].data,
        time=vis["time"].data,
        vis=vis_data,
        flags=vis_flags,
        weight=vis_weight,
        imaging_weight=vis_imaging_weight,
        integration_time=vis["integration_time"].data,
        polarisation_frame=polarisation_frame,
        source=vis.attrs["source"],
        meta=vis.attrs["meta"],
    )


def convert_blockvisibility_stokesI_to_polframe(vis, poldef=None):
    """Convert the Stokes I into full polarisation, return new Visibility

    :param vis: blockvisibility
    :param poldef: desired polarisation frame
    :return: Converted visibility data.
    """
    if vis.blockvisibility_acc.polarisation_frame == poldef:
        return vis

    npol = poldef.npol

    stokesvis = vis.blockvisibility_acc.flagged_vis[..., 0][..., numpy.newaxis]
    vis_data = numpy.repeat(stokesvis, npol, axis=-1)

    stokesflags = vis.flags.data[..., 0][..., numpy.newaxis]
    vis_flags = numpy.repeat(stokesflags, npol, axis=-1)

    stokesweight = vis.blockvisibility_acc.flagged_weight[..., 0][..., numpy.newaxis]
    vis_weight = numpy.repeat(stokesweight, npol, axis=-1)

    stokesimaging_weight = vis.blockvisibility_acc.flagged_imaging_weight[..., 0][
        ..., numpy.newaxis
    ]
    vis_imaging_weight = numpy.repeat(stokesimaging_weight, npol, axis=-1)

    vis_data[..., 1] = 0.0
    vis_data[..., 2] = 0.0

    return BlockVisibility(
        frequency=vis.frequency.data,
        channel_bandwidth=vis.channel_bandwidth.data,
        phasecentre=vis.phasecentre,
        baselines=vis["baselines"],
        configuration=vis.attrs["configuration"],
        uvw=vis["uvw"].data,
        time=vis["time"].data,
        vis=vis_data,
        flags=vis_flags,
        weight=vis_weight,
        imaging_weight=vis_imaging_weight,
        integration_time=vis["integration_time"].data,
        polarisation_frame=poldef,
        source=vis.attrs["source"],
        meta=vis.attrs["meta"],
    )
