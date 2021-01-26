.. _rascil_apps_rascil_imager:

.. py:currentmodule:: rascil.apps

.. toctree::
   :maxdepth: 3

=============
rascil_imager
=============

rascil_imager is a command line app written using RASCIL. It supports three ways of making an image:

  - invert: Inverse Fourier Transform of the visibilities to make a dirty image (or point spread function)
  - cip: The SKA Continuum Imaging Pipeline.
  - ical: The SKA Iterative Calibration Pipeline (ICAL)

Notable features:

  - Reads CASA MeasurementSets
  - Image size can be a composite of 2, 3, 5
  - Distributed across processors using Dask
  - Multi Frequency Synthesis Multiscale CLEAN available, also with distribution over facets
  - Wide field imaging using the fast and accurate nifty gridder
  - Selfcalibration available for atmosphere (T), complex gains (G), and bandpass (B)

CLI arguments are grouped:

  - :code:`--mode` prefixed parameters controls which algorithm is run.
  - :code:`--imaging` prefixed parameters control the details of the imaging such as number of pixels, cellsize
  - :code:`--clean` prefixed parameters control the clean deconvolutions (active only for modes cip and ical)
  - :code:`--calibration` prefixed parameters control the calibration in the ICAL pipeline. (active only for mode ical)
  - :code:`--dask` prefixed parameters control the use of Dask/rsexecute for distributing the processing

Example:
++++++++

The following runs the cip on a data set from the CASA examples::

    #!/bin/bash
    # Run this in the directory containing SNR_G55_10s.calib.ms
    python $RASCIL/rascil/apps/rascil_imager.py --mode cip \
    --ingest_msname SNR_G55_10s.calib.ms --ingest_dd 0 1 2 3 --ingest_vis_nchan 64 \
    --ingest_chan_per_blockvis 8 --ingest_average_blockvis True \
    --imaging_npixel 1280 --imaging_cellsize 3.878509448876288e-05 \
    --imaging_weighting robust --imaging_robustness -0.5 \
    --clean_nmajor 5 --clean_algorithm mmclean --clean_scales 0 6 10 30 60 \
    --clean_fractional_threshold 0.3 --clean_threshold 0.12e-3 --clean_nmoment 5 \
    --clean_psf_support 640 --clean_restored_output integrated


Command line arguments
++++++++++++++++++++++

.. argparse::
   :filename: ../../rascil/apps/rascil_imager.py
   :func: cli_parser
   :prog: rascil_imager.py
