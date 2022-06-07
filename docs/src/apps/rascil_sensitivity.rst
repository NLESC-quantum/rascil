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

    python $RASCIL/rascil/apps/rascil_sensitivity.py --results range_0.5_int_20 --time_range -0.25 0.25 \
        --integration_time 20 --msfile range_0.5_int_20.ms


produces the output::
Final results:
   weighting  robustness  taper  cleanbeam_bmaj  cleanbeam_bmin  cleanbeam_bpa  ...      pss_casa reltonat_casa            sa           sbs            tb     sbs_casa
0    uniform         0.0    0.0        0.000124        0.000106       0.348636  ...  5.055773e-08      4.214877  5.290084e-12  4.844478e+06  7.435200e+13  9557.074591
1     robust        -2.0    0.0        0.000125        0.000107       0.346705  ...  4.907281e-08      4.091084  5.423607e-12  4.528158e+06  8.096404e+13  9048.003290
2     robust        -1.5    0.0        0.000138        0.000119       0.366295  ...  4.237805e-08      3.532957  6.570541e-12  2.905383e+06  1.339994e+14  6449.703859
3     robust        -1.0    0.0        0.000220        0.000209      19.006936  ...  3.168845e-08      2.641790  1.669975e-11  6.384441e+05  4.295821e+14  1897.540277
4     robust        -0.5    0.0        0.000328        0.000316      40.826795  ...  2.208990e-08      1.841582  3.703912e-11  1.701715e+05  1.229183e+15   596.393758
5     robust         0.0    0.0        0.000454        0.000437      33.235117  ...  1.618849e-08      1.349596  7.111900e-11  5.956637e+04  2.721061e+15   227.625391
6     robust         0.5    0.0        0.000600        0.000577      30.284717  ...  1.360183e-08      1.133952  1.242658e-10  2.643972e+04  4.523710e+15   109.457521
7     robust         1.0    0.0        0.000729        0.000702    -149.373492  ...  1.228866e-08      1.024476  1.836020e-10  1.549264e+04  6.035397e+15    66.930950
8     robust         1.5    0.0        0.000791        0.000761      30.715780  ...  1.200501e-08      1.000829  2.160325e-10  1.241103e+04  6.792939e+15    55.570408
9     robust         2.0    0.0        0.000802        0.000772    -149.271796  ...  1.199519e-08      1.000010  2.221125e-10  1.194877e+04  6.932970e+15    54.005008
10   natural         0.0    0.0        0.000804        0.000773    -149.270574  ...  1.199506e-08      1.000000  2.228600e-10  1.189396e+04  6.950160e+15    53.823312

[11 rows x 24 columns]

Command line arguments
++++++++++++++++++++++

.. argparse::
   :filename: ../../rascil/apps/rascil_sensitivity.py
   :func: cli_parser
   :prog: rascil_sensitivity.py
