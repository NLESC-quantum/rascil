.. _rascil_apps_continuum_imaging_checker:

.. py:currentmodule:: rascil.apps

.. toctree::
   :maxdepth: 3

=============
continuum_imaging_checker
=============

continuum_imaging_checker is a command line app written using RASCIL. It uses PyBDSF to find sources in an image and check with the original inputs. Currently it features the following:

  - Reads FITS images
  - Finds sources above a certain threshold and outputs the catalog
  - Apply primary beams to the flux
  - Compares with input source catalog
 
.. code-block:: none

    usage: ci_imaging_checker.py [-h] [--ingest_fitsname INGEST_FITSNAME]
                                 [--finder_bmaj FINDER_BMAJ]
                                 [--finder_bmin FINDER_BMIN]
                                 [--finder_pos_angle FINDER_POS_ANGLE]
                                 [--finder_th_isl FINDER_TH_ISL]
                                 [--finder_th_pix FINDER_TH_PIX]
                                 [--apply_primary APPLY_PRIMARY]
                                 [--telescope_model TELESCOPE_MODEL]
                                 [--match_sep MATCH_SEP]
                                 [--source_file SOURCE_FILE]
                                 [--logfile LOGFILE]                                 
   
    optional arguments:
      -h, --help            show this help message and exit
      --ingest_fitsname INGEST_FITSNAME    FITS file to be read
      --finder_bmaj FINDER_BMAJ    Major axis of the restoring beam
      --finder_bmin FINDER_BMIN    Minor axis of the restoring beam
      --finder_pos_angle FINDER_POS_ANGLE    Positioning angle of the restoring beam
      --finder_th_isl FINDER_TH_ISL    Threshold to determine the size of the islands
      --finder_th_pix FINDER_TH_PIX    Threshold to detect source (peak value)
      --apply_primary APPLY_PRIMARY    Whether to apply primary beams
      --telescope_model TELESCOPE_MODEL    The telescope to generate primary beam correction
      --match_sep MATCH_SEP    Maximum separation in radians for the source matching
      --source_file SOURCE_FILE    Name of output source file
      --logfile LOGFILE    Name of output log file

Example:
++++++++

The following runs the a data set from the RASCIL test::

    #!/bin/bash
    # Run this in the directory containing test-imaging-pipeline-dask_continuum_imaging_restored.fits
    python $RASCIL/rascil/apps/ci_imaging_checker.py \
    --ingest_fitsname test-imaging-pipeline-dask_continuum_imaging_restored.fits


Command line arguments
++++++++++++++++++++++

.. argparse::
   :filename: ../../rascil/apps/ci_imaging_checker.py
   :func: cli_parser
   :prog: ci_imaging_checker.py

