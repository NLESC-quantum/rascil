.. _rascil_apps_continuum_imaging_checker:

.. py:currentmodule:: rascil.apps

.. toctree::
   :maxdepth: 3

=========================
continuum_imaging_checker
=========================

continuum_imaging_checker is a command line app written using RASCIL. It uses the python package PyBDSF (https://github.com/lofar-astron/PyBDSF.git) to find sources in an image and check with the original inputs. Currently it features the following:

  - Reads FITS images
  - Finds sources above a certain threshold and outputs the catalogue (in csv, fits and skycomponents format)
  - Apply a primary beam to the fluxes
  - Optional: compares with input source catalogue (takes hdf5 format)
 
.. code-block:: none

    usage: ci_imaging_checker.py [-h] [--ingest_fitsname INGEST_FITSNAME]
                                 [--finder_beam_maj FINDER_BEAM_MAJ]
                                 [--finder_beam_min FINDER_BEAM_MIN]
                                 [--finder_beam_pos_angle FINDER_BEAM_POS_ANGLE]
                                 [--finder_th_isl FINDER_TH_ISL]
                                 [--finder_th_pix FINDER_TH_PIX]
                                 [--apply_primary APPLY_PRIMARY]
                                 [--telescope_model TELESCOPE_MODEL]
                                 [--check_source CHECK_SOURCE]
                                 [--input_source_format INPUT_SOURCE_FORMAT]
                                 [--input_source_filename INPUT_SOURCE_FILENAME]
                                 [--match_sep MATCH_SEP]
                                 [--source_file SOURCE_FILE]
                                 [--logfile LOGFILE]                                 
   
    optional arguments:
      -h, --help            show this help message and exit
      --ingest_fitsname INGEST_FITSNAME    FITS file to be read
      --finder_beam_maj FINDER_BEAM_MAJ    Major axis of the restoring beam
      --finder_beam_min FINDER_BEAM_MIN    Minor axis of the restoring beam
      --finder_beam_pos_angle FINDER_BEAM_POS_ANGLE    Positioning angle of the restoring beam
      --finder_th_isl FINDER_TH_ISL    Threshold to determine the size of the islands
      --finder_th_pix FINDER_TH_PIX    Threshold to detect source (peak value)
      --apply_primary APPLY_PRIMARY    Whether to apply primary beam
      --telescope_model TELESCOPE_MODEL    The telescope to generate primary beam correction
      --check_source CHECK_SOURCE       Option to check with original input source catalogue
      --input_source_format INPUT_SOURCE_FORMAT         The input format of the source catalogue
      --input_source_filename INPUT_SOURCE_FILENAME  If use external source file, the file name of source file
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

