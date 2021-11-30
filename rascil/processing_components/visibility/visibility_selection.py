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

log = logging.getLogger("rascil-logger")


def blockvisibility_select_uv_range(bvis, uvmin=0.0, uvmax=numpy.inf):
    """Flag in-place all visibility data outside uvrange uvmin, uvmax (wavelengths)

    The flags are set to 1 for all data outside the specified uvrange

    :param bvis: BlockVisibility
    :param uvmin: Minimum uv to flag
    :param uvmax: Maximum uv to flag
    :return: bvis (with flags applied)
    """
    if uvmax is not None and uvmax < numpy.inf:
        bvis["flags"] = xarray.where(bvis["uvdist_lambda"] > uvmax, bvis["flags"], 1.0)
    if uvmin is not None and uvmin > 0.0:
        bvis["flags"] = xarray.where(bvis["uvdist_lambda"] < uvmin, bvis["flags"], 1.0)
    return bvis


def blockvisibility_select_r_range(bvis, rmin=0.0, rmax=numpy.inf):
    """Select a block visibility with stations in a range of distance from the array centre

    r is the distance from the array centre in metres

    :param bvis: BlockVisibility
    :param rmax: Maximum r
    :param rmin: Minimum r
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
