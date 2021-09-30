This directory contains scripts for simulation and processing of a small SKA observation.

- Run ska_simulate_skymodel_rsexecute_pipeline.py to generate
the simulated data. These are stored as 8 single frequency MeasurementSet to allow parallel reading 
in the subsequent scripts.
- ska_invert_skymodel_rsexecute_pipeline.py makes a dirty image.
- ska_continuum_imaging_skymodel_rsexecute_pipeline.py runs the continuum imaging pipeline. No calibration is applied.
- ska_ical_skymodel_rsexecute_pipeline.py runs the Iterative Calibration pipeline, ICAL, which performs 
  self-calibration during the imaging.

In the simulation, phase errors are applied to the visibility data. As a consequence the continuum imaging pipeline 
image has relatively poor sensitivity. The sensitivity is improved greatly in the ICAL pipeline.

Each script uses Dask, and runs for some minutes. Dask provides diagnostic
pages on port http://127.0.0.1:8787. This includes information on the entire processing (status),
the workers, the timeline of the workers, the task graph, and a graph showing groups of task.

The images are stored as FITS files and can be viewed with any FITS viewer (e.g. casaviewer or
CARTA). The gaintable is stored in an HDF file and can be viewed using e.g. HDFVIEW
