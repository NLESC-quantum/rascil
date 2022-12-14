""" Functions for calibration, including creation of gaintables, application of gaintables, and
merging gaintables.

"""

__all__ = [
    "gaintable_summary",
    "qa_gaintable",
    "apply_gaintable",
    "append_gaintable",
    "create_gaintable_from_blockvisibility",
    "create_gaintable_from_rows",
    "copy_gaintable",
    "gaintable_plot",
    "multiply_gaintables",
    "concatenate_gaintables",
]

import copy
import logging
from typing import Union

import xarray
import matplotlib.pyplot as plt
import numpy.linalg

# from astropy.visualization import time_support
from astropy.time import Time

from rascil.data_models.memory_data_models import GainTable, BlockVisibility, QA
from rascil.data_models.polarisation import ReceptorFrame
from rascil.data_models import get_parameter

log = logging.getLogger("rascil-logger")


def apply_gaintable(
    vis: BlockVisibility, gt: GainTable, inverse=False, **kwargs
) -> BlockVisibility:
    """Apply a gain table to a block visibility

    The corrected visibility is::

        V_corrected = {g_i * g_j^*}^-1 V_obs

    If the visibility data are polarised e.g. polarisation_frame("linear") then the inverse operator
    represents an actual inverse of the gains.

    :param vis: blockvisibility to have gains applied
    :param gt: Gaintable to be applied
    :param inverse: Apply the inverse (default=False)
    :return: input vis with gains applied

    """
    # assert isinstance(vis, BlockVisibility), "vis is not a BlockVisibility: %r" % vis
    # assert isinstance(gt, GainTable), "gt is not a GainTable: %r" % gt

    ntimes, nants, nchan, nrec, _ = gt.gain.shape

    if inverse:
        log.debug("apply_gaintable: Apply inverse gaintable")
    else:
        log.debug("apply_gaintable: Apply gaintable")

    if vis.blockvisibility_acc.npol == 1:
        log.debug("apply_gaintable: scalar gains")

    # row_numbers = numpy.array(list(range(len(vis.time))), dtype='int')
    row_numbers = numpy.arange(len(vis.time))

    for row in range(ntimes):
        vis_rows = (
            numpy.abs(vis.time.data - gt.time.data[row]) < gt.interval.data[row] / 2.0
        )
        vis_rows = row_numbers[vis_rows]
        if len(vis_rows) > 0:

            # Lookup the gain for this set of visibilities
            gain = gt["gain"].data[row]
            cgain = numpy.conjugate(gt["gain"].data[row])
            gainwt = gt["weight"].data[row]

            # The shape of the mueller matrix is
            nant, nchan, nrec, _ = gain.shape
            baselines = vis.baselines.data

            # Try to ignore visibility flags in application of gains. Should have no impact
            # and will save time in applying the flags
            use_flags = get_parameter(kwargs, "use_flags", False)
            flagged = use_flags and numpy.max(vis["flags"][vis_rows].data) > 0.0
            if flagged:
                log.debug("apply_gaintable:Applying flags")
                original = vis.blockvisibility_acc.flagged_vis[vis_rows]
                applied = copy.deepcopy(original)
                appliedwt = copy.deepcopy(
                    vis.blockvisibility_acc.flagged_weight[vis_rows]
                )
            else:
                log.debug("apply_gaintable:flags are absent or being ignored")
                original = vis["vis"].data[vis_rows]
                applied = copy.deepcopy(original)
                appliedwt = copy.deepcopy(vis["weight"].data[vis_rows])

            if vis.blockvisibility_acc.npol == 1:
                if inverse:
                    # lgain = numpy.ones_like(gain)
                    # lgain[numpy.abs(gain) > 0.0] = 1.0 / gain[numpy.abs(gain) > 0.0]
                    lgain = numpy.zeros_like(gain)
                    try:
                        numpy.putmask(lgain, numpy.abs(gain) > 0.0, 1.0 / gain)
                    except FloatingPointError:
                        pass
                else:
                    lgain = gain

                # tlgain = lgain.T
                # tclgain = numpy.conjugate(tlgain)
                # smueller = numpy.ones([nchan, nant, nant], dtype='complex')
                # for chan in range(nchan):
                #     smueller[chan, :, :] = numpy.ma.outer(tlgain[0, 0, chan, :],
                #                                           tclgain[0, 0, chan, :]).reshape([nant, nant])
                # numpy.testing.assert_allclose(smueller,smueller1,rtol=1e-5)

                # Original Code with Loop
                # for sub_vis_row in range(original.shape[0]):
                #     for chan in range(nchan):
                #         applied[sub_vis_row, :, :, chan, 0] = \
                #             original[sub_vis_row, :, :, chan, 0] * smueller[chan, :, :]
                #         antantwt = numpy.outer(gainwt[:, chan, 0, 0], gainwt[:, chan, 0, 0])
                #         appliedwt[sub_vis_row, :, :, chan, 0] = antantwt
                #         applied[sub_vis_row, :, :, chan, 0][antantwt == 0.0] = 0.0

                # Optimized (SIM-423)
                # smueller1 = numpy.ones([nchan, nant, nant], dtype='complex')
                smueller1 = numpy.einsum(
                    "ijlm,kjlm->jik", lgain, numpy.conjugate(lgain)
                )

                for sub_vis_row in range(original.shape[0]):
                    for ibaseline, (a1, a2) in enumerate(baselines):
                        for chan in range(nchan):
                            if numpy.abs(smueller1[chan, a1, a2]) > 0.0:
                                applied[sub_vis_row, ibaseline, chan, 0] = (
                                    original[sub_vis_row, ibaseline, chan, 0]
                                    * smueller1[chan, a1, a2]
                                )
                                appliedwt[sub_vis_row, ibaseline, chan, 0] = (
                                    gainwt[a1, chan, 0, 0] * gainwt[a2, chan, 0, 0]
                                )
                            else:
                                applied[sub_vis_row, ibaseline, chan, 0] = 0.0
                                appliedwt[sub_vis_row, ibaseline, chan, 0] = 0.0

                # smueller1 = numpy.einsum('ijlm,kjlm->ikj', lgain, numpy.conjugate(lgain))
                # for sub_vis_row in range(original.shape[0]):
                #     applied[sub_vis_row, :, :, :, 0] = \
                #         original[sub_vis_row, :, :, :, 0] * smueller1[:, :, :]
                #     antantwt = numpy.einsum('ik,jk->ijk',gainwt[:, :, 0, 0], gainwt[:, :, 0, 0])
                #     appliedwt[sub_vis_row, :, :, :, 0] = antantwt
                #     numpy.putmask(applied[sub_vis_row, :, :, :, 0], antantwt[:,:,:] == 0.0, 0.0)

            elif vis.blockvisibility_acc.npol == 2:
                has_inverse_ant = numpy.zeros([nant, nchan], dtype="bool")
                if inverse:
                    igain = gain.copy()
                    cigain = cgain.copy()
                    for a1 in range(nants):
                        for chan in range(nchan):
                            try:
                                igain[a1, chan, :, :] = numpy.linalg.inv(
                                    gain[a1, chan, :, :]
                                )
                                cigain[a1, chan, :, :] = numpy.conjugate(
                                    igain[a1, chan, :, :]
                                )
                                has_inverse_ant[a1, chan] = True
                            except numpy.linalg.LinAlgError:
                                has_inverse_ant[a1, chan] = False

                    for sub_vis_row in range(original.shape[0]):
                        for ibaseline, (a1, a2) in enumerate(baselines):
                            for chan in range(nchan):
                                if (
                                    has_inverse_ant[a1, chan]
                                    and has_inverse_ant[a2, chan]
                                ):
                                    cfs = numpy.diag(
                                        original[sub_vis_row, ibaseline, chan, ...]
                                    )
                                    applied[
                                        sub_vis_row, ibaseline, chan, ...
                                    ] = numpy.diag(
                                        igain[a1, chan, :, :]
                                        @ cfs
                                        @ cigain[a2, chan, :, :]
                                    ).reshape(
                                        [2]
                                    )
                                else:
                                    applied[sub_vis_row, ibaseline, chan, 0] = 0.0
                                    appliedwt[sub_vis_row, ibaseline, chan, 0] = 0.0

                else:
                    for sub_vis_row in range(original.shape[0]):
                        for ibaseline, (a1, a2) in enumerate(baselines):
                            for chan in range(nchan):
                                cfs = numpy.diag(
                                    original[sub_vis_row, ibaseline, chan, ...]
                                )
                                applied[sub_vis_row, ibaseline, chan, ...] = numpy.diag(
                                    gain[a1, chan, :, :] @ cfs @ cgain[a2, chan, :, :]
                                ).reshape([2])

            elif vis.blockvisibility_acc.npol == 4:
                has_inverse_ant = numpy.zeros([nant, nchan], dtype="bool")
                if inverse:
                    igain = gain.copy()
                    cigain = cgain.copy()
                    for a1 in range(nants):
                        for chan in range(nchan):
                            try:
                                igain[a1, chan, :, :] = numpy.linalg.inv(
                                    gain[a1, chan, :, :]
                                )
                                cigain[a1, chan, :, :] = numpy.conjugate(
                                    igain[a1, chan, :, :]
                                )
                                has_inverse_ant[a1, chan] = True
                            except numpy.linalg.LinAlgError:
                                has_inverse_ant[a1, chan] = False

                    for sub_vis_row in range(original.shape[0]):
                        for ibaseline, baseline in enumerate(baselines):
                            for chan in range(nchan):
                                if (
                                    has_inverse_ant[baseline[0], chan]
                                    and has_inverse_ant[baseline[1], chan]
                                ):
                                    cfs = original[
                                        sub_vis_row, ibaseline, chan, ...
                                    ].reshape([2, 2])
                                    applied[sub_vis_row, ibaseline, chan, ...] = (
                                        igain[baseline[0], chan, :, :]
                                        @ cfs
                                        @ cigain[baseline[1], chan, :, :]
                                    ).reshape([4])
                                else:
                                    applied[sub_vis_row, ibaseline, chan, ...] = 0.0
                                    appliedwt[sub_vis_row, ibaseline, chan, ...] = 0.0
                else:
                    for sub_vis_row in range(original.shape[0]):
                        for ibaseline, baseline in enumerate(baselines):
                            for chan in range(nchan):
                                cfs = original[
                                    sub_vis_row, ibaseline, chan, ...
                                ].reshape([2, 2])
                                applied[sub_vis_row, ibaseline, chan, ...] = (
                                    gain[baseline[0], chan, :, :]
                                    @ cfs
                                    @ cgain[baseline[1], chan, :, :]
                                ).reshape([4])

            else:
                times = Time(vis.time / 86400.0, format="mjd", scale="utc")
                log.warning(
                    "No row in gaintable for visibility row, time range  {} to {}".format(
                        times[0].isot, times[-1].isot
                    )
                )

            vis["vis"].data[vis_rows] = applied
            vis["weight"].data[vis_rows] = appliedwt

    return vis


def gaintable_summary(gt: GainTable):
    """Return string summarizing the Gaintable

    :param gt: Gaintable
    :returns: string

    """
    return "%.3f GB" % (gt.gaintable_acc.size())


def create_gaintable_from_blockvisibility(
    vis: BlockVisibility,
    timeslice=None,
    frequencyslice: float = None,
    jones_type="T",
    **kwargs,
) -> GainTable:
    """Create gain table from visibility.

    This makes an empty gain table consistent with the BlockVisibility.

    :param vis: BlockVisibilty
    :param timeslice: Time interval between solutions (s)
    :param frequencyslice: Frequency solution width (Hz) (NYI)
    :param jones_type: Type of calibration matrix T or G or B
    :return: GainTable

    """
    # assert isinstance(vis, BlockVisibility), "vis is not a BlockVisibility: %r" % vis

    nants = vis.blockvisibility_acc.nants

    # Set up times
    if timeslice == "auto" or timeslice is None or timeslice <= 0.0:
        utimes = vis.time
    else:
        nbins = max(
            1,
            numpy.ceil(
                (numpy.max(vis.time.data) - numpy.min(vis.time.data)) / timeslice
            ).astype("int"),
        )

        utimes = [
            numpy.average(times)
            for time, times in vis.time.groupby_bins("time", nbins, squeeze=False)
        ]
        utimes = numpy.array(utimes)

    gain_interval = numpy.ones_like(utimes)
    if len(utimes) > 1:
        for time_index, _ in enumerate(utimes):
            if time_index == 0:
                gain_interval[0] = utimes[1] - utimes[0]
            else:
                gain_interval[time_index] = utimes[time_index] - utimes[time_index - 1]

    # Set the frequency sampling
    if jones_type == "B":
        ufrequency = vis.frequency.data
        nfrequency = len(ufrequency)
    elif jones_type == "G" or jones_type == "T":
        ufrequency = numpy.average(vis.frequency) * numpy.ones([1])
        nfrequency = 1
    else:
        raise ValueError(f"Unknown Jones type {jones_type}")

    ntimes = len(utimes)

    receptor_frame = ReceptorFrame(vis.blockvisibility_acc.polarisation_frame.type)
    nrec = receptor_frame.nrec

    gainshape = [ntimes, nants, nfrequency, nrec, nrec]
    gain = numpy.ones(gainshape, dtype="complex")
    if nrec > 1:
        gain[..., 0, 1] = 0.0
        gain[..., 1, 0] = 0.0

    gain_weight = numpy.ones(gainshape)
    gain_time = utimes
    gain_frequency = ufrequency
    gain_residual = numpy.zeros([ntimes, nfrequency, nrec, nrec])

    gt = GainTable(
        gain=gain,
        time=gain_time,
        interval=gain_interval,
        weight=gain_weight,
        residual=gain_residual,
        frequency=gain_frequency,
        receptor_frame=receptor_frame,
        phasecentre=vis.phasecentre,
        configuration=vis.configuration,
        jones_type=jones_type,
    )

    # assert isinstance(gt, GainTable), "gt is not a GainTable: %r" % gt

    return gt


def append_gaintable(gt: GainTable, othergt: GainTable) -> GainTable:
    """Append othergt to gt

    :param gt:
    :param othergt:
    :return: GainTable gt + othergt
    """
    assert gt.receptor_frame == othergt.receptor_frame
    gt.data = numpy.hstack((gt.data, othergt.data))
    return gt


def copy_gaintable(gt: GainTable, zero=False):
    """Copy a GainTable

    Performs a deepcopy of the data array
    """

    if gt is None:
        return gt

    ##assert isinstance(gt, GainTable), gt

    newgt = gt.copy(deep=True)
    if zero:
        newgt["gain"].data[...] = 0.0
    return newgt


def create_gaintable_from_rows(
    gt: GainTable, rows: numpy.ndarray, makecopy=True
) -> Union[GainTable, None]:
    """Create a GainTable from selected rows

    :param gt: GainTable
    :param rows: Boolean array of row selection
    :param makecopy: Make a deep copy (True)
    :return: GainTable
    """

    if rows is None or numpy.sum(rows) == 0:
        return None

    assert (
        len(rows) == gt.ntimes
    ), "Length of rows does not agree with length of GainTable"

    # assert isinstance(gt, GainTable), gt

    if makecopy:
        newgt = copy_gaintable(gt)
        newgt.data = copy.deepcopy(gt.data[rows])
        return newgt
    else:
        gt.data = copy.deepcopy(gt.data[rows])

        return gt


def qa_gaintable(gt: GainTable, context=None) -> QA:
    """Assess the quality of a gaintable

    :param gt:
    :return: QA
    """
    if not numpy.max(gt.weight.data) > 0.0:
        raise ValueError("qa_gaintable: All gaintable weights are zero")

    agt = numpy.abs(gt.gain.data[gt.weight.data > 0.0])
    pgt = numpy.angle(gt.gain.data[gt.weight.data > 0.0])
    rgt = gt.residual.data[numpy.sum(gt.weight.data, axis=1) > 0.0]
    data = {
        "shape": gt.gain.shape,
        "maxabs-amp": numpy.max(agt),
        "minabs-amp": numpy.min(agt),
        "rms-amp": numpy.std(agt),
        "medianabs-amp": numpy.median(agt),
        "maxabs-phase": numpy.max(pgt),
        "minabs-phase": numpy.min(pgt),
        "rms-phase": numpy.std(pgt),
        "medianabs-phase": numpy.median(pgt),
        "residual": numpy.max(rgt),
    }
    return QA(origin="qa_gaintable", data=data, context=context)


def gaintable_plot(
    gt: GainTable,
    cc="T",
    title="",
    ants=None,
    channels=None,
    label_max=0,
    min_amp=1e-5,
    cmap="rainbow",
    **kwargs,
):
    """Standard plot of gain table

    :param gt: Gaintable
    :param cc: Type of gain table e.g. 'T', 'G, 'B'
    :param value: 'amp' or 'phase' or 'residual'
    :param ants: Antennas to plot
    :param channels: Channels to plot
    :param kwargs:
    :return:
    """

    if ants is None:
        ants = range(gt.gaintable_acc.nants)
    if channels is None:
        channels = range(gt.gaintable_acc.nchan)

    if gt.configuration is not None:
        labels = [gt.configuration.names[ant] for ant in ants]
    else:
        labels = ["" for ant in ants]

    # with time_support(format = 'iso', scale = 'utc'):
    # time_axis = Time(gt.time/86400.0, format='mjd', out_subfmt='str')
    time_axis = gt["time"].data / 86400.0
    ntimes = len(time_axis)
    nants = gt.gaintable_acc.nants
    nchan = gt.gaintable_acc.nchan

    if cc == "B":

        fig, ax = plt.subplots(3, 1, sharex=True)

        residual = gt["residual"].data[:, channels, 0, 0]
        ax[0].imshow(residual, cmap=cmap)
        ax[0].set_title("{title} RMS residual {cc}".format(title=title, cc=cc))
        ax[0].set_ylabel("RMS residual (Jy)")

        amp = numpy.abs(
            gt["gain"].data[:, :, channels, 0, 0].reshape([ntimes * nants, nchan])
        )
        ax[1].imshow(amp, cmap=cmap)
        ax[1].set_ylabel("Amplitude")
        ax[1].set_title("{title} Amplitude {cc}".format(title=title, cc=cc))
        ax[1].xaxis.set_tick_params(labelsize="small")

        phase = numpy.angle(
            gt["gain"].data[:, :, channels, 0, 0].reshape([ntimes * nants, nchan])
        )
        ax[2].imshow(phase, cmap=cmap)
        ax[2].set_ylabel("Phase (radian)")
        ax[2].set_title("{title} Phase {cc}".format(title=title, cc=cc))
        ax[2].xaxis.set_tick_params(labelsize="small")

    else:

        fig, ax = plt.subplots(3, 1, sharex=True)

        residual = gt["residual"].data[:, channels, 0, 0]
        ax[0].plot(time_axis, residual, ".")
        ax[1].set_ylabel("Residual fit (Jy)")
        ax[0].set_title("{title} Residual {cc}".format(title=title, cc=cc))

        for ant in ants:
            amp = numpy.abs(gt["gain"].data[:, ant, channels, 0, 0])
            ax[1].plot(
                time_axis[amp[:, 0] > min_amp],
                amp[amp[:, 0] > min_amp],
                ".",
                label=labels[ant],
            )
        ax[1].set_ylabel("Amplitude (Jy)")
        ax[1].set_title("{title} Amplitude {cc}".format(title=title, cc=cc))

        for ant in ants:
            amp = numpy.abs(gt["gain"].data[:, ant, channels, 0, 0])
            angle = numpy.angle(gt["gain"].data[:, ant, channels, 0, 0])
            ax[2].plot(
                time_axis[amp[:, 0] > min_amp],
                angle[amp[:, 0] > min_amp],
                ".",
                label=labels[ant],
            )
        ax[2].set_ylabel("Phase (rad)")
        ax[2].set_title("{title} Phase {cc}".format(title=title, cc=cc))
        ax[2].xaxis.set_tick_params(labelsize=8)
        plt.xticks(rotation=0)

        if gt.configuration is not None:
            if len(gt.configuration.names.data) < label_max:
                ax[1].legend()
                ax[1][1].legend()


def multiply_gaintables(
    gt: GainTable, dgt: GainTable, time_tolerance=1e-3
) -> GainTable:
    """Multiply two gaintables

    Returns gt * dgt

    :param gt:
    :param dgt:
    :return:
    """
    # assert isinstance(gt, GainTable), "gt is not a GainTable: %r" % gt
    # assert isinstance(dgt, GainTable), "gtdgt is not a GainTable: %r" % dgt

    # Test if times align
    mismatch = numpy.max(numpy.abs(gt["time"].data - dgt["time"].data))
    if mismatch > time_tolerance:
        raise ValueError(
            f"Gaintables not aligned in time: max mismatch {mismatch} seconds"
        )
    if dgt.gaintable_acc.nrec == gt.gaintable_acc.nrec:
        if dgt.gaintable_acc.nrec == 2:
            gt["gain"].data = numpy.einsum(
                "...ik,...ij->...kj", gt["gain"].data, dgt["gain"].data
            )
            gt["weight"].data *= dgt["weight"].data
        elif dgt.gaintable_acc.nrec == 1:
            gt["gain"].data *= dgt["gain"].data
            gt["weight"].data *= dgt["weight"].data
        else:
            raise ValueError(
                "Gain tables have illegal structures {} {}".format(str(gt), str(dgt))
            )

    else:
        raise ValueError(
            "Gain tables have different structures {} {}".format(str(gt), str(dgt))
        )

    return gt


def concatenate_gaintables(gt_list, dim="time"):
    """Concatenate a list of gaintables

    :param gt_list: List of gaintables
    :return: Concatendated gaintable
    """

    if len(gt_list) == 0:
        raise ValueError("GainTable list is empty")

    try:
        return xarray.concat(
            gt_list, dim=dim, data_vars="minimal", coords="minimal", compat="override"
        )

    except TypeError:
        # RASCIL-defined classes that inherit from xarray.Dataset, do not
        # take attrs and an input argument; xarray.concat tries to call the
        # subclass with attrs arg, because it assumes that the subclass takes
        # the same args as the Dataset class; we need to manually accommodate for this
        return xarray.concat(
            [dataset.to_native_dataset() for dataset in gt_list],
            dim=dim,
            data_vars="minimal",
            coords="minimal",
            compat="override",
        )
