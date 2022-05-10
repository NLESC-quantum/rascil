.. _rascil_apps_rascil_sensitivity:

.. py:currentmodule:: rascil.apps

.. toctree::
   :maxdepth: 3

==================
rascil_sensitivity
==================

rascil_sensitivity is a command line app written using RASCIL. It allows calculation of
point source sensitivity (pss) and surface brightness sensitivity (sbs). The analysis is
based on Dan Briggs's PhD thesis https://casa.nrao.edu/Documents/Briggs-PhD.pdf

rascil_sensitivity works by constructing a
BlockVisibility set and running invert to obtain the point spread function. The visibility weights
in the BlockVisibility are constructed to be equal to the time-bandwidth product each visibility
sample. For natural weighting, these weights are used as the imaging weights. The sum of gridded weights
therefore gives the total time-bandwidth of the observation. Given Tsys and efficiency it can then calculate the
point source sensitivity. To obtain the surface brightness sensitivity, we calculate the solid angle of
the clean beam fitted to the PSF, and divide the point source sensitivity by the solid angle.

Weighting schemes such as robust weighting and visibility tapering modify the imaging weights. The point source
sensitivity always worsens compared to natural weighting but the surface brightness sensitivity may improve.

The robustness parameter and the visibility taper can be specified as single values or as a list of values
to test.

The array configuration is specified by 2 parameters:
`configuration` identifies a table with details of the available dishes, `subarray`
names a json file listing the ids (i.e. row numbers in the configuration table) 
of the dishes to be used. If no subarray is specified then all dishes will be selected. The
json format is:: 

    {"ids": [64, 65, 66, 67, 68, 69, 70, ....etc.]}


The principal output is a CSV file, written by pandas in which all values of robustness and taper are
tested, along with natural weighting.

The processing is distributed using Dask over all frequency channels specified.

Example script
++++++++++++++

The following::

    python $RASCIL/rascil/apps/rascil_sensitivity.py --imaging_cellsize 2e-7 --imaging_npixel 1024 \
    --imaging_weighting robust --imaging_robustness -2 -1 0 1 2 --rmax 1e5 --imaging_taper 6e-7

produces the output::

    Final results:
      weighting  robustness         taper  cleanbeam_bmaj  cleanbeam_bmin  cleanbeam_bpa  ... psf_medianabsdevmedian    psf_median       pss            sa           sbs            tb
  weighting  robustness         taper  cleanbeam_bmaj  cleanbeam_bmin  ...  psf_median           pss            sa           sbs            tb
0    robust        -2.0  6.000000e-07        0.000092        0.000090  ...    0.000016  6.533630e-06  2.891386e-12  2.259688e+06  1.143939e+15
1    robust        -1.0  6.000000e-07        0.000137        0.000129  ...    0.000083  2.829928e-06  6.484715e-12  4.363997e+05  6.097629e+15
2    robust         0.0  6.000000e-07        0.000244        0.000222  ...    0.004725  1.416867e-06  2.055279e-11  6.893796e+04  2.432503e+16
3    robust         1.0  6.000000e-07        0.000364        0.000330  ...    0.148570  1.019947e-06  4.571603e-11  2.231050e+04  4.694143e+16
4    robust         2.0  6.000000e-07        0.000397        0.000361  ...    0.232532  9.457183e-07  5.449319e-11  1.735480e+04  5.459943e+16    6   natural         0.0  6.000000e-07        0.000398        0.000361     -87.531892  ...           2.594258e+13  1.283383e+14  0.000008  5.469040e-11  1.506087e+05  1.097033e+15

[5 rows x 21 columns]

Command line arguments
++++++++++++++++++++++

.. argparse::
   :filename: ../../rascil/apps/rascil_sensitivity.py
   :func: cli_parser
   :prog: rascil_sensitivity.py
