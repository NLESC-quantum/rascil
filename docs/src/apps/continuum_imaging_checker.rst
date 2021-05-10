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
  - Finds sources above a certain threshold and outputs the catalogue (in CSV, FITS and skycomponents format). For multi-frequency images, the source detection cna be performed on the central channel or average over all channels. 
  - Produces image statistics and diagnostic plots including: running mean plots of the residual, restored, background and sources and a histogram with fitted Gaussian and a power spectrum of the residual are also plotted.
  - Optional: Read in the sensitivity image and apply a primary beam correction to the fluxes.
  - Optional: compares with input source catalogue : takes hdf5 and txt format. The source input should has columns of "RA(deg), Dec(deg), FluxI(Jy), FluxQ(Jy), FluxU(Jy), FluxV(Jy), Ref. Freq.(Hz), Spectral Index".
  - Optional: plot the comparison and error of positions and fluxes for input and output source catalogue.

Example:
++++++++

The following runs the a data set from the RASCIL test::

    #!/bin/bash
    # Run this in the directory containing both the
    # restored and residual fits files:
    python $RASCIL/rascil/apps/ci_checker_main.py \
    --ingest_fitsname_restored test-imaging-pipeline-dask_continuum_imaging_restored.fits \
    --ingest_fitsname_residual test-imaging-pipeline-dask_continuum_imaging_residual.fits

If a source check is required::

    #!/bin/bash
    # This example deals with the multi-frequency image 
    python $RASCIL/rascil/apps/ci_checker_main.py \
    --ingest_fitsname_restored test-imaging-pipeline-dask_continuum_imaging_restored_cube.fits \
    --check_source True --plot_source True --input_source_format external \
     --input_source_filename test-imaging-pipeline-dask_continuum_imaging_components.hdf

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

    python ci_checker_main.py @args.txt

Specifying the ``@`` sign in front of the file name will let the code know that you want
to ready the arguments from a file instead of directly from the command line.

What happens when the image files, the argument file, and the continuum_imagin_checker code
are not all in the same directory? Let's take the following directory structure as an example::

    - rascil # this is the root directory of the RASCIL git repository
        - rascil
            - apps   
                ci_checker_main.py
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
    --plot_source=True

And you need to provide similarily the relative or absolute path both to the args file and
the code you are running::

    python rascil/apps/ci_checker_main.py @rascil/args.txt

Docker image
++++++++++++

A Docker image is available at ``nexus.engageska-portugal.pt/rascil-docker/rascil-ci-checker``
which can be run with either Docker or Singularity. Instructions can be found at

 .. toctree::
    :maxdepth: 1

    ../installation/RASCIL_docker

under **Running the continuum_imaging_checker** section.

Output plots
++++++++++++
A list of plots are generated to analyze the image as well as comparing the input and output source catelogues. 

Plots for restored image::

    ..._restored_plot.png  # Running mean of restored image
    ..._sources_plot.png  # Running mean of the sources
    ..._background_plot.png  # Running mean of background
    ..._restored_power_spectrum.png # Power spectrum of restored image

Plots for residual image::

    ..._residual_hist.png  # Histogram and Gaussian fit of residual image
    ..._residual_power_spectrum.png  # Power spectrum of residual image

Plots for position matching::

    ..._position_value.png  # RA, Dec values of input and output sources
    ..._position_error.png  # RA, Dec error (output-input)
    ..._position_distance.png # RA, Dec error with respect to distance from the centre

Plots for wide field accuracy::

    ..._position_quiver.png  # Quiver plot of the movement of source positions
    ..._gaussian_beam_position.png  # Gaussian fitted beam sizes for output sources

Plots for flux matching::

    ..._flux_value.png  # Values of output flux vs. input flux of sources
    ..._flux_ratio.png  # Ratio of flux out/flux in
    ..._flux_histogram.png  # Histogram of flux comparison
 
Plots for spectral index::

    ..._spec_index.png # Spectral index of input vs output fluxes over frequency.

Command line arguments
++++++++++++++++++++++

.. argparse::
   :filename: ../../rascil/apps/ci_checker_main.py
   :func: cli_parser
   :prog: ci_checker_main.py
