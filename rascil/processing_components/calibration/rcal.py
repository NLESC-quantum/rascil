""" Real time calibration pipeline

"""

__all__ = ["rcal"]

import collections

from rascil.data_models.memory_data_models import BlockVisibility, GainTable
from rascil.processing_components.visibility.base import copy_visibility
from rascil.processing_components.calibration.solvers import solve_gaintable


def rcal(vis: BlockVisibility, components, **kwargs) -> GainTable:
    """Real-time calibration pipeline.

    Reads visibilities through a BlockVisibility iterator, calculates model visibilities according to a
    component-based sky model, and performs calibration solution, writing a gaintable for each chunk of
    visibilities.

    :param vis: blockvisibility or Union(Visibility, Iterable)
    :param components: Component-based sky model
    :param kwargs: Parameters
    :return: gaintable
    """

    if not isinstance(vis, collections.abc.Iterable):
        vis = [vis]

    # Import here to avoid a circular dependency
    from rascil.processing_components.imaging.dft import dft_skycomponent_visibility

    for time, vischunk in vis.groupby("time", squeeze=False):
        vispred = copy_visibility(vischunk, zero=True)
        vispred = dft_skycomponent_visibility(vispred, components)
        gt = solve_gaintable(vischunk, vispred, **kwargs)
        yield gt
