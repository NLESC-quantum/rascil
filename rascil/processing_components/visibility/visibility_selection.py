"""Visibility selection functions

Visibility selection can be done using xarray capabilities. For example, for flag all long
baselines::

        bvis["flags"] = xarray.where(
            bvis["uvdist_lambda"] > 20000.0, bvis["flags"], 1.0
        )

To select by row number::

        selected_bvis = bvis.isel({"time": slice(5, 7)})
        
To select by frequency channel::

        selected_bvis = bvis.isel({"frequency": slice(1, 3)})

To select by frequency::

        selected_bvis = bvis.sel({"frequency": slice(0.9e8, 1.2e8)})
        
To select by frequency and polarisation::

        selected_bvis = bvis.sel(
            {"frequency": slice(0.9e8, 1.2e8), "polarisation": ["XX", "YY"]}
        ).dropna(dim="frequency", how="all")

In addition there are rascil functions which wrap up more complex selections.
To flag all data in uvrange uvmin, uvmax (wavelengths)::

    bvis = blockvisibility_flag_uvrange(bvis, uvmin, uvmax)

Note that this is not a selection operator but flags the unwanted data instead.

To select all data with dishes/stations with distance from the array
centre in a range rmin, rmax (metres)::
    
    selected_bvis = blockvisibility_select_r_range(bvis, rmin, rmax)

"""

__all__ = ["blockvisibility_select_uv_range", "blockvisibility_select_r_range"]
import logging
import numpy
import xarray
import pandas

log = logging.getLogger("rascil-logger")


def blockvisibility_select_uv_range(bvis, uvmin=0.0, uvmax=1.0e15):
    """Flag in-place all visibility data outside uvrange uvmin, uvmax (wavelengths)

    The flags are set to 1 for all data outside the specified uvrange

    :param bvis: BlockVisibility
    :param uvmin: Minimum uv to flag
    :param uvmax: Maximum uv to flag
    :return: bvis (with flags applied)
    """
    if uvmax is not None:
        bvis["flags"] = xarray.where(bvis["uvdist_lambda"] < uvmax, bvis["flags"], 1)
    if uvmin is not None:
        bvis["flags"] = xarray.where(bvis["uvdist_lambda"] > uvmin, bvis["flags"], 1)
    return bvis


def blockvisibility_select_r_range(bvis, rmin=None, rmax=None):
    """Select a block visibility with stations in a range of distance from the array centre

    r is the distance from the array centre in metres

    :param bvis: BlockVisibility
    :param rmax: Maximum r
    :param rmin: Minimum r
    :return: Selected BlockVisibility
    """
    if rmin is None and rmax is None:
        return bvis

    # Calculate radius from array centre (in 3D) and set it as a data variable
    xyz0 = bvis.configuration.xyz - bvis.configuration.xyz.mean("id")
    r = numpy.sqrt(xarray.dot(xyz0, xyz0, dims="spatial"))
    config = bvis.configuration.assign(radius=r)
    # Now use it for selection
    if rmax is None:
        sub_config = config.where(config["radius"] > rmin, drop=True)
    elif rmin is None:
        sub_config = config.where(config["radius"] < rmax, drop=True)
    else:
        sub_config = config.where(config["radius"] > rmin, drop=True).where(
            config["radius"] < rmax, drop=True
        )

    ids = list(sub_config.id.data)
    baselines = bvis.baselines.where(
        bvis.baselines.antenna1.isin(ids), drop=True
    ).where(bvis.baselines.antenna2.isin(ids), drop=True)
    sub_bvis = bvis.sel({"baselines": baselines}, drop=True)

    # The baselines coord now is missing the antenna1, antenna2 keys
    # so we add those back
    def generate_baselines(id):
        for a1 in id:
            for a2 in id:
                if a2 >= a1:
                    yield a1, a2

    sub_bvis["baselines"] = pandas.MultiIndex.from_tuples(
        generate_baselines(ids),
        names=("antenna1", "antenna2"),
    )
    return sub_bvis
