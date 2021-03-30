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

What happens when the image files, the argument file, and the continuum_imagin_checker code
are not all in the same directory? Let's take the following directory structure as an example::

    - rascil # this is the root directory of the RASCIL git repository
        - rascil
            - apps
                ci_imaging_checker.py
            - my_data
                my_restored_file.fits
                my_residual_file.fits
            args.txt

With such a setup, the best way to run the checker is from the top-level ``rascil`` directory
(the git root directory). Your ``args.txt`` file will need to contain either the relative or
absolute path to your FITS files. E.g.::

    --ingest_fitsname_restored=rascil/my_data/test-imaging-pipeline-dask_continuum_imaging_restored.fits
    --ingest_fitsname_residual=rascil/my_data/test-imaging-pipeline-dask_continuum_imaging_residual.fits
    --check_source=True

And you need to provide similarily the relative or absolute path both to the args file and
the code you are running::

    python rascil/apps/ci_imaging_checker.py @rascil/args.txt

Docker image
++++++++++++

A Docker image is available at ``nexus.engageska-portugal.pt/rascil-docker/rascil-ci-checker``
which can be run with either Docker or Singularity. Instructions can be found at

 .. toctree::
    :maxdepth: 1

    ../installation/RASCIL_docker

under **Running the continuum_imaging_checker** section.

Command line arguments
++++++++++++++++++++++

.. argparse::
   :filename: ../../rascil/apps/ci_imaging_checker.py
   :func: cli_parser
   :prog: ci_imaging_checker.py
