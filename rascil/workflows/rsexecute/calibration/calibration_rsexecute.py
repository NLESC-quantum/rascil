"""

"""

__all__ = ["calibrate_list_rsexecute_workflow"]

import logging

from rascil.processing_components.calibration import (
    apply_calibration_chain,
    solve_calibrate_chain,
)
from rascil.processing_components.visibility import (
    integrate_visibility_by_channel,
    divide_visibility,
)
from rascil.processing_components.visibility import concatenate_visibility

from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute

log = logging.getLogger("rascil-logger")


def calibrate_list_rsexecute_workflow(
    vis_list,
    model_vislist,
    gt_list=None,
    calibration_context="TG",
    controls=None,
    global_solution=True,
    **kwargs,
):
    """Create a set of components for (optionally global) calibration of a list of visibilities

    If global solution is true then visibilities are gathered to a single visibility data set which is then
    self-calibrated. The resulting gaintable is then effectively scattered out for application to each visibility
    set. If global solution is false then the solutions are performed locally.

    :param vis_list: list of visibilities (or graph)
    :param model_vislist: list of model visibilities (or graph)
    :param calibration_context: String giving terms to be calibrated e.g. 'TGB'
    :param controls: Calibration controls dictionary
    :param global_solution: Solve for global gains
    :param kwargs: Parameters for functions in components
    :return: list of calibrated vis, list of dictionaries of gaintables
    """

    def calibration_solve(vis, modelvis=None, gt=None, do_global=None):
        if do_global:
            log.info(
                "calibration_solve: Performing global solution of gains for all blockvis"
            )
        else:
            log.info(
                "calibration_solve: Performing seperate solution of gains foe each blockvis"
            )

        return solve_calibrate_chain(
            vis,
            modelvis,
            gt,
            calibration_context=calibration_context,
            controls=controls,
            **kwargs,
        )

    def calibration_apply(vis, gt):
        assert gt is not None
        return apply_calibration_chain(
            vis,
            gt,
            calibration_context=calibration_context,
            controls=controls,
            **kwargs,
        )

    # Here we do a global solution over all blockvis and channels or
    # just solutions per blockvis
    if global_solution and (len(vis_list) > 1):
        # The conversion is a no op if it's actually a blockvis
        point_vislist = [
            rsexecute.execute(divide_visibility, nout=1)(vis_list[i], model_vislist[i])
            for i, _ in enumerate(vis_list)
        ]

        global_point_vis_list = rsexecute.execute(concatenate_visibility, nout=1)(
            point_vislist, dim="frequency"
        )
        # global_point_vis_list = rsexecute.execute(
        #     integrate_visibility_by_channel, nout=1
        # )(global_point_vis_list)
        # This is a global solution so we only compute one gain table
        if gt_list is None or len(gt_list) < 1:
            gt_list = [
                rsexecute.execute(calibration_solve, pure=True, nout=1)(
                    global_point_vis_list,
                    do_global=global_solution,
                )
            ]
        else:
            gt_list = [
                rsexecute.execute(calibration_solve, pure=True, nout=1)(
                    global_point_vis_list,
                    gt=gt_list[0],
                    do_global=True,
                )
            ]

        return [
            rsexecute.execute(calibration_apply, nout=1)(v, gt_list[0])
            for v in vis_list
        ], gt_list
    else:
        if gt_list is not None and len(gt_list) > 0:
            gt_list = [
                rsexecute.execute(calibration_solve, pure=True, nout=1)(
                    v,
                    model_vislist[i],
                    gt_list[i],
                    do_global=False,
                )
                for i, v in enumerate(vis_list)
            ]
        else:
            gt_list = [
                rsexecute.execute(calibration_solve, pure=True, nout=1)(
                    v, model_vislist[i]
                )
                for i, v in enumerate(vis_list)
            ]
        return [
            rsexecute.execute(calibration_apply)(v, gt_list[i])
            for i, v in enumerate(vis_list)
        ], gt_list
