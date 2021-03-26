.. _rascil_apps_continuum_imaging_checker:

.. py:currentmodule:: rascil.apps

.. toctree::
   :maxdepth: 3

=========================
continuum_imaging_checker
=========================

continuum_imaging_checker is a command line app written using RASCIL.
It uses the python package `PyBDSF <https://github.com/lofar-astron/PyBDSF.git>`_ to find sources in an image
and check with the original inputs. Currently it features the following:

  - Reads FITS images.
  - Finds sources above a certain threshold and outputs the catalogue (in CSV, FITS and skycomponents format).
  - Produces residual image statistics and plots a histogram of the noise with Gaussian fit.
  - Optional: apply a primary beam to the fluxes.
  - Optional: compares with input source catalogue : takes hdf5 and txt format. The source input should has columns of "RA(deg), Dec(deg), Flux(Jy)".

.. code-block:: none

    usage: ci_imaging_checker.py [-h] [--ingest_fitsname_restored INGEST_FITSNAME_RESTORED]
                                 [--ingest_fitsname_residual INGEST_FITSNAME_RESIDUAL]
                                 [--finder_beam_maj FINDER_BEAM_MAJ]
                                 [--finder_beam_min FINDER_BEAM_MIN]
                                 [--finder_beam_pos_angle FINDER_BEAM_POS_ANGLE]
                                 [--finder_th_isl FINDER_TH_ISL]
                                 [--finder_th_pix FINDER_TH_PIX]
                                 [--apply_primary APPLY_PRIMARY]
                                 [--telescope_model TELESCOPE_MODEL]
                                 [--check_source CHECK_SOURCE]
                                 [--plot_source PLOT_SOURCE]
                                 [--input_source_format INPUT_SOURCE_FORMAT]
                                 [--input_source_filename INPUT_SOURCE_FILENAME]
                                 [--match_sep MATCH_SEP]
                                 [--quiet_bdsf  QUIET_BDSF]
                                 [--source_file SOURCE_FILE]
                                 [--rascil_source_file RASCIL_SOURCE_FILE]
                                 [--logfile LOGFILE]

    optional arguments:
      -h, --help            show this help message and exit
      --ingest_fitsname_restored INGEST_FITSNAME_RESTORED    restored FITS file to be read
      --ingest_fitsname_residual INGEST_FITSNAME_RESIDUAL    residual FITS file to be read
      --finder_beam_maj FINDER_BEAM_MAJ    Major axis of the restoring beam
      --finder_beam_min FINDER_BEAM_MIN    Minor axis of the restoring beam
      --finder_beam_pos_angle FINDER_BEAM_POS_ANGLE    Positioning angle of the restoring beam
      --finder_th_isl FINDER_TH_ISL    Threshold to determine the size of the islands
      --finder_th_pix FINDER_TH_PIX    Threshold to detect source (peak value)
      --apply_primary APPLY_PRIMARY    Whether to apply primary beam
      --telescope_model TELESCOPE_MODEL    The telescope to generate primary beam correction
      --check_source CHECK_SOURCE       Option to check with original input source catalogue
      --plot_source PLOT_SOURCE         Option to plot position and flux errors for source catalogue
      --input_source_format INPUT_SOURCE_FORMAT         The input format of the source catalogue
      --input_source_filename INPUT_SOURCE_FILENAME  If use external source file, the file name of source file
      --match_sep MATCH_SEP    Maximum separation in radians for the source matching
      --quiet_bdsf  QUIET_BDSF    If True, suppress bdsf.process_image() text output to screen. Output is still sent to the log file.
      --source_file SOURCE_FILE    Name of output source file
      --rascil_source_file RASCIL_SOURCE_FILE    Name of output RASCIL skycomponents file
      --logfile LOGFILE    Name of output log file

Supplying arguments from a file:
++++++++++++++++++++++++++++++++

You can also load arguments into the app from a file.

Example arguments file, called `args.txt`::

    --ingest_fitsname_restored=test-imaging-pipeline-dask_continuum_imaging_restored.fits
    --ingest_fitsname_residual=test-imaging-pipeline-dask_continuum_imaging_residual.fits
    --check_source=True
    --plot_source=True

Make sure each line contains one argument, there is an equal sign between arg and its value,
and that there aren't any trailing white spaces in the lines.

Then run the checker as follows::

    python ci_imaging_checker.py @args.txt

Specifying the ``@`` sign in front of the file name will let the code know that you want
to ready the arguments from a file instead of directly from the command line.

Example:
++++++++

The following runs the a data set from the RASCIL test::

    #!/bin/bash
    # Run this in the directory containing both the
    # restored and residual fits files:
    # test-imaging-pipeline-dask_continuum_imaging_restored.fits
    # test-imaging-pipeline-dask_continuum_imaging_residual.fits
    python $RASCIL/rascil/apps/ci_imaging_checker.py \
    --ingest_fitsname_restored test-imaging-pipeline-dask_continuum_imaging_restored.fits \
    --ingest_fitsname_residual test-imaging-pipeline-dask_continuum_imaging_residual.fits

If a source check is required::

    #!/bin/bash
    python $RASCIL/rascil/apps/ci_imaging_checker.py \
    --ingest_fitsname_restored test-imaging-pipeline-dask_continuum_imaging_restored.fits \
    --ingest_fitsname_residual test-imaging-pipeline-dask_continuum_imaging_residual.fits \
    --check_source True --plot_source True\
    --input_source_format external --input_source_filename $RASCIL/data/models/GLEAM_filtered.txt

Docker image
++++++++++++

A Docker image is available at ``nexus.engageska-portugal.pt/rascil-docker/rascil-ci-checker``
which can be run with either Docker or Singularity. Instructions can be found at

 .. toctree::
    ../installation/RASCIL_docker

under **Running the continuum_imaging_checker** section.

Command line arguments
++++++++++++++++++++++

.. argparse::
   :filename: ../../rascil/apps/ci_imaging_checker.py
   :func: cli_parser
   :prog: ci_imaging_checker.py
