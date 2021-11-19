""" Functions to solve for antenna/station gain

This uses an iterative substitution algorithm due to Larry D'Addario c 1980'ish. Used
in the original VLA Dec-10 Antsol.


For example::

    gtsol = solve_gaintable(vis, originalvis, phase_only=True, niter=niter, crosspol=False, tol=1e-6)
    vis = apply_gaintable(vis, gtsol, inverse=True)
 

"""

__all__ = ["solve_gaintable"]

import logging

import numpy

from rascil.data_models.memory_data_models import BlockVisibility, GainTable
from rascil.processing_components.calibration.operations import (
    create_gaintable_from_blockvisibility,
)
from rascil.processing_components.visibility.operations import divide_visibility

log = logging.getLogger("rascil-logger")


def solve_gaintable(
    vis: BlockVisibility,
    modelvis: BlockVisibility = None,
    gt=None,
    phase_only=True,
    niter=200,
    tol=1e-6,
    crosspol=False,
    normalise_gains=True,
    jones_type="T",
    **kwargs,
) -> GainTable:
    """Solve a gain table by fitting an observed visibility to a model visibility

    If modelvis is None, a point source model is assumed.

    :param vis: BlockVisibility containing the observed data_models
    :param modelvis: BlockVisibility containing the visibility predicted by a model
    :param gt: Existing gaintable
    :param phase_only: Solve only for the phases (default=True)
    :param niter: Number of iterations (default 30)
    :param tol: Iteration stops when the fractional change in the gain solution is below this tolerance
    :param crosspol: Do solutions including cross polarisations i.e. XY, YX or RL, LR
    :param jones_type: Type of calibration matrix T or G or B
    :return: GainTable containing solution

    """
    # assert isinstance(vis, BlockVisibility), vis
    if modelvis is not None:
        # assert isinstance(modelvis, BlockVisibility), modelvis
        assert numpy.max(numpy.abs(modelvis.vis)) > 0.0, "Model visibility is zero"

    if phase_only:
        log.debug("solve_gaintable: Solving for phase only")
    else:
        log.debug("solve_gaintable: Solving for complex gain")

    if gt is None:
        log.debug("solve_gaintable: creating new gaintable")
        gt = create_gaintable_from_blockvisibility(vis, jones_type=jones_type, **kwargs)
    else:
        log.debug("solve_gaintable: starting from existing gaintable")

    if modelvis is not None:
        pointvis = divide_visibility(vis, modelvis)
    else:
        pointvis = vis

    nants = gt.gaintable_acc.nants
    nchan = gt.gaintable_acc.nchan
    npol = pointvis.blockvisibility_acc.npol
    # Do we want to average over frequency?
    if nchan == 1:
        axes = (0, 2)
    else:
        axes = 0
    for row, time in enumerate(gt.time):
        time_slice = {
            "time": slice(time - gt.interval[row] / 2, time + gt.interval[row] / 2)
        }
        pointvis_sel = pointvis.sel(time_slice)
        if pointvis_sel.blockvisibility_acc.ntimes > 0:
            x_b = numpy.sum(
                (pointvis_sel.vis.data * pointvis_sel.weight.data)
                * (1 - pointvis_sel.flags.data),
                axis=axes,
            )
            xwt_b = numpy.sum(
                pointvis_sel.weight.data * (1 - pointvis_sel.flags.data), axis=axes
            )
            x = numpy.zeros([nants, nants, nchan, npol], dtype="complex")
            xwt = numpy.zeros([nants, nants, nchan, npol])
            for ibaseline, (a1, a2) in enumerate(pointvis.baselines.data):
                x[a1, a2, ...] = x_b[ibaseline, ...]
                xwt[a1, a2, ...] = xwt_b[ibaseline, ...]
                x[a2, a1, ...] = numpy.conjugate(x_b[ibaseline, ...])
                xwt[a2, a1, ...] = xwt_b[ibaseline, ...]

            mask = numpy.abs(xwt) > 0.0
            if numpy.sum(mask) > 0:
                x_shape = x.shape
                x[mask] = x[mask] / xwt[mask]
                x[~mask] = 0.0
                xwt[mask] = xwt[mask] / numpy.max(xwt[mask])
                xwt[~mask] = 0.0
                x = x.reshape(x_shape)

                if vis.blockvisibility_acc.npol == 1:
                    (
                        gt["gain"].data[row, ...],
                        gt["weight"].data[row, ...],
                        gt["residual"].data[row, ...],
                    ) = solve_antenna_gains_itsubs_scalar(
                        gt["gain"].data[row, ...],
                        gt["weight"].data[row, ...],
                        x,
                        xwt,
                        phase_only=phase_only,
                        niter=niter,
                        tol=tol,
                    )
                elif vis.blockvisibility_acc.npol == 2:
                    (
                        gt["gain"].data[row, ...],
                        gt["weight"].data[row, ...],
                        gt["residual"].data[row, ...],
                    ) = solve_antenna_gains_itsubs_nocrossdata(
                        gt["gain"].data[row, ...],
                        gt["weight"].data[row, ...],
                        x,
                        xwt,
                        phase_only=phase_only,
                        niter=niter,
                        tol=tol,
                    )
                elif vis.blockvisibility_acc.npol == 4:
                    if crosspol:
                        (
                            gt["gain"].data[row, ...],
                            gt["weight"].data[row, ...],
                            gt["residual"].data[row, ...],
                        ) = solve_antenna_gains_itsubs_matrix(
                            gt["gain"].data[row, ...],
                            gt["weight"].data[row, ...],
                            x,
                            xwt,
                            phase_only=phase_only,
                            niter=niter,
                            tol=tol,
                        )
                    else:
                        (
                            gt["gain"].data[row, ...],
                            gt["weight"].data[row, ...],
                            gt["residual"].data[row, ...],
                        ) = solve_antenna_gains_itsubs_nocrossdata(
                            gt["gain"].data[row, ...],
                            gt["weight"].data[row, ...],
                            x,
                            xwt,
                            phase_only=phase_only,
                            niter=niter,
                            tol=tol,
                        )

                else:
                    (
                        gt["gain"].data[row, ...],
                        gt["weight"].data[row, ...],
                        gt["residual"].data[row, ...],
                    ) = solve_antenna_gains_itsubs_scalar(
                        gt["gain"].data[row, ...],
                        gt["weight"].data[row, ...],
                        x,
                        xwt,
                        phase_only=phase_only,
                        niter=niter,
                        tol=tol,
                    )

                if normalise_gains and not phase_only:
                    gabs = numpy.average(numpy.abs(gt["gain"].data[row]))
                    gt["gain"].data[row] /= gabs
            else:
                gt["gain"].data[row, ...] = 1.0 + 0.0j
                gt["weight"].data[row, ...] = 0.0
                gt["residual"].data[row, ...] = 0.0

        else:
            log.warning(
                "Gaintable {0}, vis time mismatch {1}".format(gt.time, vis.time)
            )

    # assert isinstance(gt, GainTable), "gt is not a GainTable: %r" % gt

    return gt


def solve_antenna_gains_itsubs_scalar(
    gain, gwt, x, xwt, niter=200, tol=1e-6, phase_only=True, refant=0, damping=0.5
):
    """Solve for the antenna gains

    x(antenna2, antenna1) = gain(antenna1) conj(gain(antenna2))

    This uses an iterative substitution algorithm due to Larry
    D'Addario c 1980'ish (see ThompsonDaddario1982 Appendix 1). Used
    in the original VLA Dec-10 Antsol.

    :param gain: gains
    :param gwt: gain weight
    :param x: Equivalent point source visibility[nants, nants, ...]
    :param xwt: Equivalent point source weight [nants, nants, ...]
    :param niter: Number of iterations
    :param tol: tolerance on solution change
    :param phase_only: Do solution for only the phase? (default True)
    :param refant: Reference antenna for phase (default=0)
    :return: gain [nants, ...], weight [nants, ...]

    """

    nants = x.shape[0]
    # Optimized
    i_diag = numpy.diag_indices(nants, nants)
    x[i_diag[0], i_diag[1], ...] = 0.0
    xwt[i_diag[0], i_diag[1], ...] = 0.0
    i_lower = numpy.tril_indices(nants, -1)
    i_upper = (i_lower[1], i_lower[0])
    x[i_upper] = numpy.conjugate(x[i_lower])
    xwt[i_upper] = xwt[i_lower]
    # Original
    # for ant1 in range(nants):
    #     x[ant1, ant1, ...] = 0.0
    #     xwt[ant1, ant1, ...] = 0.0
    #     for ant2 in range(ant1 + 1, nants):
    #         x[ant1, ant2, ...] = numpy.conjugate(x[ant2, ant1, ...])
    #         xwt[ant1, ant2, ...] = xwt[ant2, ant1, ...]

    for iter in range(niter):
        gainLast = gain
        gain, gwt = gain_substitution_scalar(gain, x, xwt)
        if phase_only:
            mask = numpy.abs(gain) > 0.0
            gain[mask] = gain[mask] / numpy.abs(gain[mask])
        angles = numpy.angle(gain)
        gain *= numpy.exp(-1j * angles)[refant, ...]
        gain = (1.0 - damping) * gain + damping * gainLast
        change = numpy.max(numpy.abs(gain - gainLast))
        if change < tol:
            if phase_only:
                mask = numpy.abs(gain) > 0.0
                gain[mask] = gain[mask] / numpy.abs(gain[mask])
            return gain, gwt, solution_residual_scalar(gain, x, xwt)

    log.warning(
        "solve_antenna_gains_itsubs_scalar: gain solution failed, retaining gain solutions"
    )

    if phase_only:
        mask = numpy.abs(gain) > 0.0
        gain[mask] = gain[mask] / numpy.abs(gain[mask])

    return gain, gwt, solution_residual_scalar(gain, x, xwt)


def gain_substitution_scalar(gain, x, xwt):
    nants, nchan, nrec, _ = gain.shape
    # newgain = numpy.ones_like(gain, dtype='complex128')
    # gwt = numpy.zeros_like(gain, dtype='double')
    newgain1 = numpy.ones_like(gain, dtype="complex128")
    gwt1 = numpy.zeros_like(gain, dtype="double")

    xxwt = x * xwt[:, :, :]
    cgain = numpy.conjugate(gain)
    gcg = gain[:, :] * cgain[:, :]
    # Optimzied
    n_top = numpy.einsum("ik...,ijk...->jk...", gain, xxwt)
    n_bot = numpy.einsum("ik...,ijk...->jk...", gcg, xwt).real

    # Convert mask to putmask
    # mask = n_bot > 0.0
    # notmask = n_bot <= 0.0
    # newgain1[:, :][mask] = n_top[mask] / n_bot[mask]
    # newgain1[:, :][notmask] = 0.0
    numpy.putmask(newgain1, n_bot > 0.0, n_top / n_bot)
    numpy.putmask(newgain1, n_bot <= 0.0, 0.0)

    gwt1[:, :] = n_bot
    # gwt1[:, :][notmask] = 0.0
    numpy.putmask(gwt1, n_bot <= 0.0, 0.0)

    newgain1 = newgain1.reshape([nants, nchan, nrec, nrec])
    gwt1 = gwt1.reshape([nants, nchan, nrec, nrec])
    return newgain1, gwt1
    # Original scripts
    # for ant1 in range(nants):
    #     ntt = gain[:, :, 0, 0] * xxwt[:, ant1, :]
    #     top = numpy.sum(gain[:, :, 0, 0] * xxwt[:, ant1, :], axis=0)
    #     bot = numpy.sum((gcg[:, :] * xwt[:, ant1, :, 0, 0]).real, axis=0)
    #
    #     if bot.all() > 0.0:
    #         newgain[ant1, :, 0, 0] = top / bot
    #         gwt[ant1, :, 0, 0] = bot
    #     else:
    #         newgain[ant1, :, 0, 0] = 0.0
    #         gwt[ant1, :, 0, 0] = 0.0
    # assert(newgain == newgain1).all()
    # assert(gwt == gwt1).all()
    # return newgain, gwt


def solve_antenna_gains_itsubs_nocrossdata(
    gain, gwt, x, xwt, niter=200, tol=1e-6, phase_only=True, refant=0
):
    """Solve for the antenna gains using full matrix expressions, but no cross hands

    x(antenna2, antenna1) = gain(antenna1) conj(gain(antenna2))

    See Appendix D, section D.1 in:

    J. P. Hamaker, “Understanding radio polarimetry - IV. The full-coherency analogue of
    scalar self-calibration: Self-alignment, dynamic range and polarimetric fidelity,” Astronomy
    and Astrophysics Supplement Series, vol. 143, no. 3, pp. 515–534, May 2000.

    :param gain: gains
    :param gwt: gain weight
    :param x: Equivalent point source visibility[nants, nants, ...]
    :param xwt: Equivalent point source weight [nants, nants, ...]
    :param niter: Number of iterations
    :param tol: tolerance on solution change
    :param phase_only: Do solution for only the phase? (default True)
    :param refant: Reference antenna for phase (default=0.0)
    :return: gain [nants, ...], weight [nants, ...]
    """

    # This implementation is sub-optimal. TODO: Reimplement IQ, IV calibration
    nants, _, nchan, npol = x.shape
    if npol == 2:
        newshape = (nants, nants, nchan, 4)
        x_fill = numpy.zeros(newshape, dtype="complex")
        x_fill[..., 0] = x[..., 0]
        x_fill[..., 3] = x[..., 1]
        xwt_fill = numpy.zeros(newshape, dtype="float")
        xwt_fill[..., 0] = xwt[..., 0]
        xwt_fill[..., 3] = xwt[..., 1]
    else:
        x_fill = x
        x_fill[..., 1] = 0.0
        x_fill[..., 2] = 0.0
        xwt_fill = xwt
        xwt_fill[..., 1] = 0.0
        xwt_fill[..., 2] = 0.0

    return solve_antenna_gains_itsubs_matrix(
        gain,
        gwt,
        x_fill,
        xwt_fill,
        niter=niter,
        tol=tol,
        phase_only=phase_only,
        refant=refant,
    )  # for ant1 in range(nants):


def solve_antenna_gains_itsubs_matrix(
    gain, gwt, x, xwt, niter=200, tol=1e-6, phase_only=True, refant=0
):
    """Solve for the antenna gains using full matrix expressions

    x(antenna2, antenna1) = gain(antenna1) conj(gain(antenna2))

    See Appendix D, section D.1 in:

    J. P. Hamaker, “Understanding radio polarimetry - IV. The full-coherency analogue of
    scalar self-calibration: Self-alignment, dynamic range and polarimetric fidelity,” Astronomy
    and Astrophysics Supplement Series, vol. 143, no. 3, pp. 515–534, May 2000.

    :param gain: gains
    :param gwt: gain weight
    :param x: Equivalent point source visibility[nants, nants, ...]
    :param xwt: Equivalent point source weight [nants, nants, ...]
    :param niter: Number of iterations
    :param tol: tolerance on solution change
    :param phase_only: Do solution for only the phase? (default True)
    :param refant: Reference antenna for phase (default=0.0)
    :return: gain [nants, ...], weight [nants, ...]
    """

    nants, _, nchan, npol = x.shape
    assert npol == 4
    newshape = (nants, nants, nchan, 2, 2)
    x = x.reshape(newshape)
    xwt = xwt.reshape(newshape)

    # Optimzied
    i_diag = numpy.diag_indices(nants, nants)
    x[i_diag[0], i_diag[1], ...] = 0.0
    xwt[i_diag[0], i_diag[1], ...] = 0.0
    i_lower = numpy.tril_indices(nants, -1)
    i_upper = (i_lower[1], i_lower[0])
    x[i_upper] = numpy.conjugate(x[i_lower])
    xwt[i_upper] = xwt[i_lower]
    # Original
    # for ant1 in range(nants):
    #     x[ant1, ant1, ...] = 0.0
    #     xwt[ant1, ant1, ...] = 0.0
    #     for ant2 in range(ant1 + 1, nants):
    #         x[ant1, ant2, ...] = numpy.conjugate(x[ant2, ant1, ...])
    #         xwt[ant1, ant2, ...] = xwt[ant2, ant1, ...]

    gain[..., 0, 1] = 0.0
    gain[..., 1, 0] = 0.0

    for iter in range(niter):
        gainLast = gain
        gain, gwt = gain_substitution_matrix(gain, x, xwt)
        if phase_only:
            mask = numpy.abs(gain) > 0.0
            gain[mask] = gain[mask] / numpy.abs(gain[mask])
        change = numpy.max(numpy.abs(gain - gainLast))
        gain = 0.5 * (gain + gainLast)
        if change < tol:
            return gain, gwt, solution_residual_matrix(gain, x, xwt)

    log.warning(
        "solve_antenna_gains_itsubs_scalar: gain solution failed, retaining gain solutions"
    )

    return gain, gwt, solution_residual_matrix(gain, x, xwt)


def gain_substitution_matrix(gain, x, xwt):
    nants, nchan, nrec, _ = gain.shape
    # newgain = numpy.ones_like(gain, dtype='complex128')
    newgain1 = numpy.ones_like(gain, dtype="complex128")
    # gwt = numpy.zeros_like(gain, dtype='double')
    gwt1 = numpy.zeros_like(gain, dtype="double")

    # We are going to work with Jones 2x2 matrix formalism so everything has to be
    # converted to that format
    x = x.reshape([nants, nants, nchan, nrec, nrec])
    diag = numpy.ones_like(x)
    xwt = xwt.reshape([nants, nants, nchan, nrec, nrec])
    # Write these loops out explicitly. Derivation of these vector equations is tedious but they are
    # structurally identical to the scalar case with the following changes
    # Vis -> 2x2 coherency vector, g-> 2x2 Jones matrix, *-> matmul, conjugate->Hermitean transpose (.H)
    #
    gain_conj = numpy.conjugate(gain)
    for ant in range(nants):
        diag[ant, ant, ...] = 0
    n_top1 = numpy.einsum("ij...->j...", xwt * diag * x * gain[:, None, ...])
    # n_top1 *= gain
    # n_top1 = numpy.conjugate(n_top1)
    n_bot = diag * xwt * gain_conj * gain
    n_bot1 = numpy.einsum("ij...->i...", n_bot)

    # Using Boolean Index - 158 ms
    # newgain1[:, :][n_bot1[:,:] > 0.0] = n_top1[n_bot1[:,:] > 0.0] / n_bot1[n_bot1[:,:] > 0.0]
    # newgain1[:,:][n_bot1[:,:] <= 0.0] = 0.0

    # Using putmask: 121 ms
    n_top2 = n_top1.copy()
    numpy.putmask(n_top2, n_bot1[...] <= 0, 0.0)
    n_bot2 = n_bot1.copy()
    numpy.putmask(n_bot2, n_bot1[...] <= 0, 1.0)
    newgain1 = n_top2 / n_bot2

    gwt1 = n_bot1.real
    return newgain1, gwt1

    # Original Scripts translated from Fortran
    #
    # for ant1 in range(nants):
    #     for chan in range(nchan):
    #         top = 0.0
    #         bot = 0.0
    #         for ant2 in range(nants):
    #             if ant1 != ant2:
    #                 xmat = x[ant2, ant1, chan]
    #                 xwtmat = xwt[ant2, ant1, chan]
    #                 g2 = gain[ant2, chan]
    #                 top += xmat * xwtmat * g2
    #                 bot += numpy.conjugate(g2) * xwtmat * g2
    #         newgain[ant1, chan][bot > 0.0] = top[bot > 0.0] / bot[bot > 0.0]
    #         newgain[ant1, chan][bot <= 0.0] = 0.0
    #         gwt[ant1, chan] = bot.real
    # assert(newgain==newgain1).all()
    # assert(gwt == gwt1).all()
    # return newgain, gwt


def solution_residual_scalar(gain, x, xwt):
    """Calculate residual across all baselines of gain for point source equivalent visibilities

    :param gain: gain [nant, ...]
    :param x: Point source equivalent visibility [nant, ...]
    :param xwt: Point source equivalent weight [nant, ...]
    :return: residual[...]
    """

    nant, nchan, nrec, _ = gain.shape
    x = x.reshape(nant, nant, nchan, nrec, nrec)

    xwt = xwt.reshape(nant, nant, nchan, nrec, nrec)

    residual = numpy.zeros([nchan, nrec, nrec])
    sumwt = numpy.zeros([nchan, nrec, nrec])

    for chan in range(nchan):
        lgain = gain[:, chan, 0, 0]
        clgain = numpy.conjugate(lgain)
        smueller = numpy.ma.outer(clgain, lgain).reshape([nant, nant])
        error = x[:, :, chan, 0, 0] - smueller
        for i in range(nant):
            error[i, i] = 0.0
        residual[chan] += numpy.sum(
            error * xwt[:, :, chan, 0, 0] * numpy.conjugate(error)
        ).real
        sumwt[chan] += numpy.sum(xwt[:, :, chan, 0, 0])

    residual[sumwt > 0.0] = numpy.sqrt(residual[sumwt > 0.0] / sumwt[sumwt > 0.0])
    residual[sumwt <= 0.0] = 0.0

    return residual


def solution_residual_matrix(gain, x, xwt):
    """Calculate residual across all baselines of gain for point source equivalent visibilities

    :param gain: gain [nant, ...]
    :param x: Point source equivalent visibility [nant, ...]
    :param xwt: Point source equivalent weight [nant, ...]
    :return: residual[...]
    """

    nants, _, nchan, nrec, _ = x.shape

    # residual = numpy.zeros([nchan, nrec, nrec])
    # sumwt = numpy.zeros([nchan, nrec, nrec])

    n_residual = numpy.zeros([nchan, nrec, nrec])
    n_sumwt = numpy.zeros([nchan, nrec, nrec])

    n_gain = numpy.einsum("i...,j...->ij...", numpy.conjugate(gain), gain)
    n_error = numpy.conjugate(x - n_gain)
    nn_residual = (n_error * xwt * numpy.conjugate(n_error)).real
    n_residual = numpy.einsum("ijk...->k...", nn_residual)
    n_sumwt = numpy.einsum("ijk...->k...", xwt)

    n_residual[n_sumwt > 0.0] = numpy.sqrt(
        n_residual[n_sumwt > 0.0] / n_sumwt[n_sumwt > 0.0]
    )
    n_residual[n_sumwt <= 0.0] = 0.0

    return n_residual

    # This is written out in long winded form but should e optimised for
    # production code!
    # for ant1 in range(nants):
    #     for ant2 in range(nants):
    #         for chan in range(nchan):
    #             for rec1 in range(nrec):
    #                 for rec2 in range(nrec):
    #                     error = x[ant2, ant1, chan, rec2, rec1] - \
    #                             gain[ant1, chan, rec2, rec1] * numpy.conjugate(gain[ant2, chan, rec2, rec1])
    #                     residual[chan, rec2, rec1] += (error * xwt[ant2, ant1, chan, rec2, rec1] * numpy.conjugate(
    #                         error)).real
    #                     sumwt[chan, rec2, rec1] += xwt[ant2, ant1, chan, rec2, rec1]
    #
    # residual[sumwt > 0.0] = numpy.sqrt(residual[sumwt > 0.0] / sumwt[sumwt > 0.0])
    # residual[sumwt <= 0.0] = 0.0
    # # assert (residual == n_residual).all()
    # return residual
