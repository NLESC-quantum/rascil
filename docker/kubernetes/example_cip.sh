python3 ${RASCIL}/rascil/apps/rascil_imager.py  --clean_nmoment 3 --clean_facets 8 --clean_overlap 32 \
  --clean_nmajor 3  --clean_niter 10000 --clean_threshold 2e-3 --clean_restore_facets 1 \
  --clean_gain 0.1 --clean_psf_support 256  --clean_algorithm mmclean \
  --use_dask True --dask_scheduler ${RASCIL_DASK_SCHEDULER} --imaging_context ng --imaging_npixel 2048 --imaging_pol stokesI \
  --clean_restored_output taylor --clean_fractional_threshold 0.1 \
  --mode cip --imaging_ng_threads 32 \
  --imaging_cellsize 4.84813681109536e-06 --imaging_weighting robust --imaging_robustness -2.0 --imaging_nchan 1 \
  --ingest_vis_nchan 16 --ingest_chan_per_blockvis 2 \
  --ingest_msname /mnt/data/test.ms --logfile tmp.log \
