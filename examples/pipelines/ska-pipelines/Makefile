
# Disable built in rules
.SUFFIXES:

MS = ska-pipeline_simulation_0.ms ska-pipeline_simulation_1.ms ska-pipeline_simulation_2.ms \
ska-pipeline_simulation_3.ms ska-pipeline_simulation_4.ms ska-pipeline_simulation_5.ms \
ska-pipeline_simulation_6.ms ska-pipeline_simulation_7.ms

FITS = ska-continuum-imaging_rsexecute_deconvolved_cube.fits \
ska-continuum-imaging_rsexecute_residual_cube.fits \
ska-continuum-imaging_rsexecute_restored_cube.fits \
ska-ical_rsexecute_deconvolved_cube.fits \
ska-ical_rsexecute_residual_cube.fits \
ska-ical_rsexecute_restored_cube.fits \
ska-imaging_rsexecute_dirty_centre.fits \
ska-imaging_rsexecute_dirty_cube.fits 

HDF = ska-ical_rsexecute_gaintable.hdf

LOG = ska-pipeline.log

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
