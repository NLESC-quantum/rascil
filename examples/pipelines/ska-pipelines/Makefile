
# Disable built in rules
.SUFFIXES:

MS = results/ska-pipeline_simulation_0.ms \
results/ska-pipeline_simulation_1.ms \
results/ska-pipeline_simulation_2.ms \
results/ska-pipeline_simulation_3.ms \
results/ska-pipeline_simulation_4.ms \
results/ska-pipeline_simulation_5.ms \
results/ska-pipeline_simulation_6.ms \
results/ska-pipeline_simulation_7.ms

FITS = results/ska-continuum-imaging_rsexecute_deconvolved_cube.fits \
results/ska-continuum-imaging_rsexecute_residual_cube.fits \
results/ska-continuum-imaging_rsexecute_restored_cube.fits \
results/ska-ical_rsexecute_deconvolved_cube.fits \
results/ska-ical_rsexecute_residual_cube.fits \
results/ska-ical_rsexecute_restored_cube.fits \
results/ska-imaging_rsexecute_dirty_centre.fits \
results/ska-imaging_rsexecute_dirty_cube.fits 

HDF = results/ska-ical_rsexecute_gaintable.hdf

LOG = results/ska-pipeline.log

.PHONY: clean
clean:
	rm -rf ${MS} ${FITS} ${HDF} ${LOG}

ms: ska_simulate_skymodel_rsexecute_pipeline.py
	time python ska_simulate_skymodel_rsexecute_pipeline.py

dirty: ms ska_invert_skymodel_rsexecute_pipeline.py
	time python ska_invert_skymodel_rsexecute_pipeline.py

cip: ms ska_continuum_imaging_skymodel_rsexecute_pipeline.py
	time python ska_continuum_imaging_skymodel_rsexecute_pipeline.py

ical: ms ska_ical_skymodel_rsexecute_pipeline.py
	time python ska_ical_skymodel_rsexecute_pipeline.py

all: ms dirty cip ical
