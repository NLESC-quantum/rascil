
NBs = bandpass-calibration.ipynb simple-dask_rsexecute.ipynb imaging.ipynb \
demo_visibility_xarray.ipynb
# demo_image_xarray.ipynb

RSTs=  $(NBs:.ipynb=.rst)

JUPYTER ?= jupyter
TIMEOUT = 3600

.PHONY: rst
rst: $(RSTs)

%.rst: %.ipynb
	jupyter nbconvert --execute --ExecutePreprocessor.timeout=$(TIMEOUT) --to rst $<

%.html: %.rst
	${RST2HTML} $(<:.rstw=.rst)  > $@

.PHONY: clean
clean:
	rm -f ${RSTs}
