.. _rascil_examples:

Examples
========

Running notebooks
*****************

The best way to get familiar with RASCIL is via jupyter notebooks. For example::

   jupyter notebook examples/notebooks/imaging.ipynb

See the jupyter notebooks below:

.. toctree::
   :maxdepth: 3

   examples/notebooks/imaging.rst
   examples/notebooks/simple-dask_rsexecute.rst
   examples/notebooks//bandpass-calibration.rst
   examples/notebooks/demo_visibility_xarray.rst

Some functions initially developed for the LOFAR telescope pipeline are made available in RASCIL. The following notebooks show how the functions are integrated.

.. toctree::
   :maxdepth: 3
   
   examples/notebooks/deconvolution.rst
   examples/notebooks/multi_frequency_deconvolution.rst

In addition, there are other notebooks in examples/notebooks that are not built as part of this documentation.
In some cases it may be necessary to add the following to the notebook to locate the RASCIL data
:code:`%env RASCIL_DATA=~/rascil_data/data`

Running scripts
***************

Some example scripts are found in the directory examples/scripts.

.. toctree::
   :maxdepth: 3

   examples/scripts/imaging.rst
   examples/scripts/primary_beam_zernikes.rst

SKA simulations
***************

* :ref:`genindex`
* :ref:`modindex`

.. _feedback: mailto:realtimcornwell@gmail.com

