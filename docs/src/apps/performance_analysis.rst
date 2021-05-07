.. _rascil_apps_performance_analysis:

.. py:currentmodule:: rascil.apps

.. toctree::
   :maxdepth: 3

====================
performance_analysis
====================

performance_analysis is a command line app written using RASCIL. It helps in analysis of performance
files written by rascil_imager.

The performance files can be obtained using a script to iterate over some parameter. For example::

    #!/usr/bin/env bash
    #
    # Run this from the directory containing the MS
    #
    results_dir=${HOME}/data/ska_mid_simulations/results/5km_resource_modelling
    for npixel in 1024 2048 4096 8192
    do
      python3 ${RASCIL}/rascil/apps/rascil_imager.py  --clean_nmoment 3 --clean_facets 8 --clean_nmajor 10 \
        --clean_threshold 3e-5 --clean_restore_facets 4 --clean_restore_overlap 32 \
        --use_dask True --imaging_context ng --imaging_npixel ${npixel} --imaging_pol stokesI --clean_restored_output list \
        --imaging_cellsize 5e-6 --imaging_weighting uniform --imaging_nchan 1 \
        --ingest_vis_nchan 100 --ingest_chan_per_blockvis 16 \
        --ingest_msname ${results_dir}/SKA_MID_SIM_custom_B2_dec_-45.0_nominal_nchan100.ms \
        --performance_file ${results_dir}/performance_rascil_imager_${npixel}.json
      mkdir ${results_dir}/5km_${npixel}
      mv ${results_dir}/SKA_MID_SIM_custom_B2_dec_-45.0_nominal_nchan100/*.fits ${results_dir}/5km_${npixel}
    done

Command line arguments
++++++++++++++++++++++

.. argparse::
   :filename: ../../rascil/apps/performance_analysis.py
   :func: cli_parser
   :prog: performance_analysis.py
