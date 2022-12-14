.. _rascil_apps_rascil_rcal:

.. py:currentmodule:: rascil.apps

.. toctree::
   :maxdepth: 3

===========
rascil_rcal
===========

rascil_rcal is a command line app written using RASCIL. It simulates the real-time
calibration pipeline RCAL. In the SKA, an initial calibration is performed in
real-time as the visibility data are accumulated. An accurate sky model is
assumed to be available or a point source model is used.

In rascil_rcal a MeasurementSet is read in and then iterated through in time-order
solving for the gains. The gaintables are accumulated into a single gain table that is written
as an HDF file.

There is also an additional plotting function that plots the gaintable values
(gain amplitude, phase and residual) over time. If plotting is required,
please make sure you have the correct path `--plot_dir` set up.
The output file name will contain the datetime of the first time sample in the data.

RFI Flagger
+++++++++++

rascil_rcal also implements reading RFI (Radio Frequency Interference) flags
and using them as part of the pipeline. Flagging is optional and can be
controlled with the `flag_rfi` argument.

RASCIL's BlockVisibility object contains a "flags" data array with the same
dimensions as the visibilities. This array is updated with the results of
the SKA Processing Function Library
`RFI Flagger <https://gitlab.com/ska-telescope/sdp/ska-sdp-func/-/blob/main/src/ska_sdp_func/rfi_flagger.py>`_,
which uses the sum-threshold method for flagging.
The RFI flagger requires `initial threshold` and `rho` values (both needed
to provide a list of thresholds used for finding RFI signal in the data), which can
be set via CLI arguments, though we recommend using the defaults at this stage.

Example script
++++++++++++++

The following runs the real time calibration pipeline on an MS generated by the
MID continuum imaging simulations (with an optional input components file)::

    #!/bin/bash
    python3 $RASCIL/rascil/apps/rascil_rcal.py \
    --ingest_msname SKA_MID_SIM_custom_B2_dec_-45.0_nominal_nchan100_actual.ms \
    --ingest_components_file SKA_MID_SIM_custom_B2_dec_-45.0_nominal_nchan100_components.hdf

There are also additional options if you want the sky model to have primary beams applied.
Currently we support internal beam from MID and LOW, or additional beam file (in FITS format).
An example::

    #!/bin/bash
    python3 $RASCIL/rascil/apps/rascil_rcal.py \
    --ingest_msname myms.ms \
    --ingest_components_file my_components.hdf \
    --apply_beam True --ingest_beam_file my_beam.fits \

Command line arguments
++++++++++++++++++++++

.. argparse::
   :filename: ../../rascil/apps/rascil_rcal.py
   :func: cli_parser
   :prog: rascil_rcal.py
