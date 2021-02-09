.. _rascil_data:

.. toctree::
   :maxdepth: 2

Data containers used by RASCIL
==============================

RASCIL holds data in python Classes. The bulk data and attributes are usually kept in a xarray.Dataset.
For each xarray based class there is an accessor which holds class specific methods and properties.

See :py:mod:`rascil.data_models.memory_data_models` for the following definitions:

* Image (data and WCS header): :py:class:`rascil.data_models.memory_data_models.Image`
* Skycomponent (data for a point source or a Gaussian source): :py:class:`rascil.data_models.memory_data_models.Skycomponent`
* SkyModel (collection of SkyComponents and Images): :py:class:`rascil.data_models.memory_data_models.SkyModel`
* Baseline-based visibility table, shape (ntimes, nbaselines, nchan, npol), length ntime): :py:class:`rascil.data_models.memory_data_models.BlockVisibility`
* Telescope Configuration: :py:class:`rascil.data_models.memory_data_models.Configuration`
* GainTable for gain solutions (as e.g. output from solve_gaintable): :py:class:`rascil.data_models.memory_data_models.GainTable`
* PointingTable for pointing information: :py:class:`rascil.data_models.memory_data_models.PointingTable`
* FlagTable for flagging information: :py:class:`rascil.data_models.memory_data_models.FlagTable`
