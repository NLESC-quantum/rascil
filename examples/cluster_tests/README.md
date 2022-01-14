
These are tests of the ability to run dask in various contexts. These are intended to be run in an initial port and
 whenever the cluster configuration has changed. The tests should be runnable from the commmand line and via SLURM
  scripts.

 - cluster_dask_test is python + Dask
 - cluster_image_test is python + Dask + RASCIL
 - ritoy is python + Dask
 
 - Numba caused a conflict in the requirements.txt. After the discussion, we removed all numba related scripts in ORC-1146. 