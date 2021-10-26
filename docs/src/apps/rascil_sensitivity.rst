.. _rascil_apps_rascil_sensitivity:

.. py:currentmodule:: rascil.apps

.. toctree::
   :maxdepth: 3

==================
rascil_sensitivity
==================

rascil_sensitivity is a command line app written using RASCIL. It allows calculation of
point source sensitivity (pss) and surface brightness sensitivity (sbs). It works by constructing a
BlockVisibility set and running invert to obtain the point spread function. Given Tsys and
efficiency it can then calculate the point source sensitivity and the surface brightness sensitivity.

The robustness parameter and the visibility taper can be specified as a list of values
to test.

The principal output is a CSV file, written by pandas.

The processing is distributed using Dask over all frequency channels specified.

Example script
++++++++++++++

The following::

    python $RASCIL/rascil/apps/rascil_sensitivity.py --imaging_cellsize 2e-7 --imaging_npixel 1024 \
    --imaging_weighting robust --imaging_robustness -2 -1 0 1 2 --rmax 1e5 --imaging_taper 6e-7

produces the results::

    Final results:
      weighting  robustness         taper  cleanbeam_bmaj  cleanbeam_bmin  cleanbeam_bpa  ... psf_medianabsdevmedian psf_median           pss            sa       sbs            tb
    0    robust        -2.0  6.000000e-07        0.000096        0.000092      71.880849  ...               0.001562   0.000054  8.730606e-14  3.166886e-12  0.027568  3.124831e+15
    1    robust        -1.0  6.000000e-07        0.000182        0.000165      92.559362  ...               0.003117   0.000817  1.189315e-14  1.141254e-11  0.001042  2.293898e+16
    2    robust         0.0  6.000000e-07        0.000307        0.000280      92.428029  ...               0.019193   0.037445  5.161449e-15  3.262469e-11  0.000158  5.285660e+16
    3    robust         1.0  6.000000e-07        0.000386        0.000351      92.337415  ...               0.043916   0.210730  3.529462e-15  5.147059e-11  0.000069  7.729695e+16
    4    robust         2.0  6.000000e-07        0.000391        0.000355      92.333318  ...               0.044735   0.221847  3.460271e-15  5.266808e-11  0.000066  7.884255e+16

    [5 rows x 21 columns]
(

Command line arguments
++++++++++++++++++++++

.. argparse::
   :filename: ../../rascil/apps/rascil_sensitivity.py
   :func: cli_parser
   :prog: rascil_sensitivity.py
