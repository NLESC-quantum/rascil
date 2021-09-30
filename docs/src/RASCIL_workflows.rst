.. _rascil_workflows:

.. toctree::
   :maxdepth: 2


Workflows
=========

Workflows coordinate processing using the data models, processing components, and processing library. These are high
level functions, and are available in an rsexecute (i.e. dask) version and sometimes a scalar version.

Calibration workflows
---------------------

* Calibrate workflow: :py:func:`rascil.workflows.rsexecute.calibration.calibrate_list_rsexecute_workflow`


Imaging workflows
-----------------

* Invert: :py:func:`rascil.workflows.rsexecute.imaging.invert_list_rsexecute_workflow` :py:func:`rascil.workflows.serial.imaging.invert_list_serial_workflow`
* Predict: :py:func:`rascil.workflows.rsexecute.imaging.predict_list_rsexecute_workflow` :py:func:`rascil.workflows.serial.imaging.predict_list_serial_workflow`
* Deconvolve: :py:func:`rascil.workflows.rsexecute.imaging.deconvolve_list_rsexecute_workflow` :py:func:`rascil.workflows.serial.imaging.deconvolve_list_serial_workflow`

Pipeline workflows
------------------

* ICAL: :py:func:`rascil.workflows.rsexecute.pipelines.ical_skymodel_list_rsexecute_workflow`
* Continuum imaging: :py:func:`rascil.workflows.rsexecute.pipelines.continuum_imaging_skymodel_list_rsexecute_workflow`
* Spectral line imaging: :py:func:`rascil.workflows.rsexecute.pipelines.spectral_line_imaging_skymodel_list_rsexecute_workflow`
* MPCCAL: :py:func:`rascil.workflows.rsexecute.pipelines.mpccal_skymodel_list_rsexecute_workflow`

Simulation workflows
--------------------

* Testing and simulation support: :py:func:`rascil.workflows.rsexecute.simulation.simulate_list_rsexecute_workflow`

Execution
---------

* Execution framework (an interface to Dask): :py:func:`rascil.workflows.rsexecute.execution_support`



