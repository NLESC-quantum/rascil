.. _RASCIL_xarray:

.. toctree::
   :maxdepth: 3

Use of xarray
*************

From release 0.2+, RASCIL has moved to use the `Xarray <https:/www.dask.org>`_ library instead of numpy in the
data classes.  RASCIL data classes are now all derived from xarray.Dataset. This change is motivated
by the large range of capababilities available from xarray. These include:

 - Named dimensions and coordinates, allowing access via quantities such as time. frequency, polarisation, receptor
 - Indexing, selection, iteration, and conditions
 - Support of split-apply-recombine operations
 - Interpolation in coordinates, including missing values
 - Automatic invocation of Dask for array operations
 - Arbitrary meta data as attributes

We have chosen to make the RASCIL data classes derive from xarray.Dataset. Instead of adding
class methods to the RASCIL data class, which would introduce some interface fragility
as xarray changes over time, we have used data accessors to control access to
methods specfic to the class. This design is suggested in the xarray documentation
on extending xarray. Examples::

    # Flagged visibility
    vis.blockvisibility_acc.flagged_vis

    # UVW in wavelengths
    vis.blockvisibility_acc.uvw_lambda

    # DataArray sizes
    vis.blockvisibility_acc.datasizes

    # Phasecentre as an astropy.SkyCoord
    im.image_acc.phasecentre

    # Image RA, Dec grid
    im.image_acc.ra_dec_mesh

    # Gaintable number of receptors
    gt.gaintable_acc.nrec

For examples of the capabilities afforded by xarray see the jupyter notebooks below:

.. toctree::
   :maxdepth: 3

   examples/notebooks/demo_image_xarray.rst
   examples/notebooks/demo_visibility_xarray.rst
