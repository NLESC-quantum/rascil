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
   examples/notebooks//bandpass-calibration_serial.rst
   examples/notebooks/demo_image_xarray.rst
   examples/notebooks/demo_visibility_xarray.rst

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

The SKA simulations make full use of the capabilities of RASCIL. The surface simulation and atmosphere simulation
both require special large data files that are not part of the repository. However, the pointing simulation can be run
using data files in the RASCIL data repository.

.. toctree::
   :maxdepth: 1

   examples/ska_simulations/mid_simulation.rst

* :ref:`genindex`
* :ref:`modindex`

.. _feedback: mailto:realtimcornwell@gmail.com

