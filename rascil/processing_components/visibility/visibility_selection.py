"""Unit tests for visibility selectors

"""

__all__ = ["blockvisibility_flag_uvrange", "blockvisibility_select_r_range"]
import logging
import numpy
import xarray

log = logging.getLogger("rascil-logger")


def blockvisibility_flag_uvrange(bvis, uvmin=0.0, uvmax=numpy.inf):
    bvis["flags"] = xarray.where(bvis["uvdist_lambda"] < uvmax, bvis["flags"], 1.0)
    bvis["flags"] = xarray.where(bvis["uvdist_lambda"] > uvmin, bvis["flags"], 1.0)
    return bvis


def blockvisibility_select_r_range(bvis, rmin=0.0, rmax=numpy.inf):
    """Select a block visibility with stations in a range of distance from the array centre

    :param bvis:
    :param rmax:
    :param rmin:
    :return: Selected BlockVisibility
    """
    # Calculate radius from array centre (in 3D) and set it as a data variable
    r = numpy.abs(bvis.configuration.xyz - bvis.configuration.xyz.mean("id")).std(
        "spatial"
    )
    config = bvis.configuration.assign(radius=r)
    # Now use it for selection
    sub_config = config.where(config["radius"] > rmin, drop=True).where(
        config["radius"] < rmax, drop=True
    )
    ids = sub_config.id.data
    return bvis.where(bvis.baselines.antenna1.isin(ids), drop=True).where(
        bvis.baselines.antenna2.isin(ids), drop=True
    )
