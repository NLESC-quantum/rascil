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

The array configuration is specified by parameters configuration and subarray.
`configuration' identifies a table with details of the available dishes. `subarray'
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
    0   uniform         0.0  6.000000e-07        0.000091        0.000089    -140.225605  ...           2.577095e+10 -1.252453e+08  0.000056  2.849622e-12  1.982420e+07  2.332254e+13
    1    robust        -2.0  6.000000e-07        0.000093        0.000092    -130.253940  ...           2.846821e+10 -1.457455e+08  0.000054  3.002486e-12  1.783327e+07  2.596078e+13
    2    robust        -1.0  6.000000e-07        0.000139        0.000130      87.455295  ...           2.680955e+11  7.178078e+09  0.000025  6.658207e-12  3.719128e+06  1.213794e+14
    3    robust         0.0  6.000000e-07        0.000244        0.000222      92.537382  ...           1.749116e+12  1.171527e+12  0.000012  2.059444e-11  6.001038e+05  4.872916e+14
    4    robust         1.0  6.000000e-07        0.000364        0.000330     -87.500856  ...           1.976384e+13  6.987381e+13  0.000009  4.574926e-11  1.944207e+05  9.407809e+14
    5    robust         2.0  6.000000e-07        0.000397        0.000361     -87.531411  ...           2.583999e+13  1.272543e+14  0.000008  5.453211e-11  1.512364e+05  1.094270e+15
    6   natural         0.0  6.000000e-07        0.000398        0.000361     -87.531892  ...           2.594258e+13  1.283383e+14  0.000008  5.469040e-11  1.506087e+05  1.097033e+15

    [7 rows x 21 columns]

Command line arguments
++++++++++++++++++++++

.. argparse::
   :filename: ../../rascil/apps/rascil_sensitivity.py
   :func: cli_parser
   :prog: rascil_sensitivity.py
