.. Data

.. toctree::
   :maxdepth: 2

Data containers used by RASCIL
==============================

RASCIL holds data in python Classes. The bulk data is usually kept in a python structured array, and the meta data as
attributes.

See :py:mod:`rascil.data_models.memory_data_models` for the following definitions:

* Image (data and WCS header): :py:class:`rascil.data_models.memory_data_models.Image`
* Skycomponent (data for a point source or a Gaussian source): :py:class:`rascil.data_models.memory_data_models.Skycomponent`
* SkyModel (collection of SkyComponents and Images): :py:class:`rascil.data_models.memory_data_models.SkyModel`
* Antenna-based visibility table, shape (nants, nants, nchan, npol), length ntime): :py:class:`rascil.data_models.memory_data_models.BlockVisibility`
* Baseline based visibility tables shape (npol,), length nvis :py:class:`rascil.data_models.memory_data_models.Visibility`
* Telescope Configuration: :py:class:`rascil.data_models.memory_data_models.Configuration`
* GainTable for gain solutions (as e.g. output from solve_gaintable): :py:class:`rascil.data_models.memory_data_models.GainTable`


