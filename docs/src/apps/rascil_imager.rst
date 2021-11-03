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

  - Reads a CASA MeasurementSet and writes FITS files
  - Image size can be a composite of 2, 3, 5
  - Distribute processing across processors using Dask
  - Multi Frequency Synthesis Multiscale CLEAN available, also with distribution of CLEAN over facets
  - Distribution of restoration over facets
  - Wide field imaging using the fast and accurate nifty gridder
  - Modelling of bright sources by fitting with sub-pixel locations
  - Selfcalibration available for atmosphere (T), complex gains (G), and bandpass (B)

CLI arguments are grouped:

  - :code:`--mode` prefixed parameters controls which algorithm is run.
  - :code:`--imaging` prefixed parameters control the details of the imaging such as number of pixels, cellsize
  - :code:`--clean` prefixed parameters control the clean deconvolutions (active only for modes cip and ical)
  - :code:`--calibration` prefixed parameters control the calibration in the ICAL pipeline. (active only for mode ical)
  - :code:`--dask` prefixed parameters control the use of Dask/rsexecute for distributing the processing

MeasurementSet ingest
+++++++++++++++++++++

Although a CASA MeasurementSet can hold heterogeneous observations, identified by data descriptors. rascil-imager can
only process identical data descriptors from a MS. The number of channels and polarisation must be the same.

Each selected data descriptor is optionally split into a number of channels optionally averaged and placed into one
BlockVisibility.

For example, using the arguments::

    --ingest_msname SNR_G55_10s.calib.ms --ingest_dd 0 1 2 3 --ingest_vis_nchan 64 \
    --ingest_chan_per_blockvis 8 --ingest_average_blockvis True

will read data descriptors 0, 1, 2, 3, each of which has 64 channels. Each set of 64 channels are split
into blocks of 8 and averaged. We thus end up with 32 separate datasets in RASCIL, each of which
is a BlockVisibility and has 1 channel, for a total of 32 channels. If the argument :code:`--ingest_average_blockvis`
is set to False, each BlockVisibility has eight channels, for a total of 256 channels.

Imaging
+++++++

To make an image from visibilities or to predict visibilities from a model, it is necessary to use a gridder.
Nifty gridder (https://gitlab.mpcdf.mpg.de/ift/nifty_gridder) is currently the best gridder to use in RASCIL.
It is written in c and uses OpenMP to distribute the processing across multiple threads.
The Nifty Gridder uses an improved wstacking algorithm uses many fewer w-planes than w stacking or
w projection. It is not necessary to explicitly set the number of w-planes.

The gridder is set by the :code:`--imaging_context` argument. The default, :code:`--imaging_context ng` is the Nifty
Gridder.

CLEAN
+++++

rascil-imager supports Hogbom CLEAN, MultiScale CLEAN, and Multi-Frequency Synthesis MultiScale Clean
(also known as MMCLEAN). The first two work independently on different frequency channels, while
MMClean works jointly cross all channels using a Taylor Series expansion in frequency for the emission.

The clean methods support a number of processing speed enhancements:

     - The multi-frequency-synthesis CLEAN works by fitting a Taylor series in frequency.
       The :code:`--ingest_chan_per_blockvis` argument controls the aggregation of channels
       in the MeasurementSet to form image planes for the CLEAN. Within a BlockVisibility the
       different channels are gridded together to form one image. Each image is then used in the
       mmclean algorithm. For example, a data set may have 256 channels spread over 4 data descriptors.
       We can split these into 32 BlockVisibilities and then run the mmclean over these 32
       channels.
     - Only a limited central region of the PSF will be subtracted during the minor cycles.
     - The cleaning may be partitioned into overlapping facets, each of which is cleaned independently,
       and then merged with neighbours using a taper function. This works well for fields of compact sources
       but is likely to not perform well for extended emission.
     - The restoration may be distributed via subimages. This requires that the subimages have significant
       overlap such that the clean beam can fit within the overlap area.

Bright compact sources can optionally be represented by discrete components instead of pixels.

 - :code:`--clean_component_threshold 0.5` All sources > 0.5 Jy to be fitted
 - :code:`--clean_component_method fit` non-linear last squares algorithm to find source parameters

The skymodel written at the end of processing will include both the image model and the
skycomponents.

Polarisation
++++++++++++

The polarisation processing behaviour is controlled by :code:`--image_pol`.

 - :code:`--image_pol stokesI` will image only the I Stokes parameter
 - :code:`--image_pol stokesIQUV` will image all Stokes parameters I, Q, U, V

Note that the combination of MM CLEAN and stokesIQUV imaging is not likely to be meaningful.

Self-calibration
++++++++++++++++

rascil-imager supports self-calibration as part of the imaging. At the end of each major cycle
a calibration solution and application may optionally be performed.

Calibration uses the Hamaker Bregman Sault formalism with the following Jones matrices supported: T (Atmospheric phase),
G (Electronics gain), B - (Bandpass).

An example consider the arguments::

    calibration_T_first_selfcal = 2
    calibration_T_phase_only = True
    calibration_T_timeslice = None
    calibration_G_first_selfcal = 5
    calibration_G_phase_only = False
    calibration_G_timeslice = 1200.0
    calibration_B_first_selfcal = 8
    calibration_B_phase_only = False
    calibration_B_timeslice = 1.0e5
    calibration_global_solution = True
    calibration_calibration_context = "TGB"

These will perform a phase only solution of the T term after the second major cycle for every integration,
solution of G after 5 major cycles with timescale of 1200s, and solution of B after 8 major cycles, integrating
across all frequencies where appropriate.

SkyModel in ICAL
++++++++++++++++

When running rascil_imager in mode ical, optionally, an initial SkyModel can be used.
To do this, set :code:`--use_initial_skymodel` to True.
The SkyModel is made up of model images (created based on input BlockVisibilities),
and SkyComponents. The kind of SkyComponent(s) to use in the initial SkyModel is controlled
by the :code:`--input_skycomponent_file` and :code:`--num_bright_sources` arguments:

1. If no input file is provided, a point source at the phase centre, with brightness of 1 Jy
   is used as the component.
2. If either an HDF file or a TXT file is provided, the components are read from the file.
    a. if :code:`--num_bright_sources` is left as :code:`None`, all of the components are used
       for the SkyModel
    b. if :code:`--num_bright_sources` is an integer `n` (`n>0`), then `n` number of
       the brightest components are used for the SkyModel

This SkyModel is then overwritten during the remaining cycles of the run.

By default, :code:`--use_initial_skymodel` is set to False, and hence no
initial SkyModel is used.

Dask
++++

Dask is used to distribute processing across multiple cores or nodes. The setup and execution of a
set of workers is controlled by a scheduler. By default, rascil uses the process scheduler which
sets up a number of processes each with a number of threads. If the host has 16 cores, the set up
will be 4 processes each with 4 threads for a total of 16 Dask workers.

For distribution across a cluster, the Dask distributed processor is required. See :ref:`RASCIL_dask`
for more details.

Example script
++++++++++++++

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
