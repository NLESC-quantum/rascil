

# Disable built in rules
.SUFFIXES:

SCs = imaging.py primary_beam_zernikes.py

TXTs=  $(SCs:.py=.txt)

.PHONY: txt
txt: $(TXTs)

%.txt: %.py
	python3 $< > $@

RSTs=  $(SCs:.py=.rst)

.PHONY: rst
rst: $(RSTs)

%.rst: %.py %.txt
	cat $@.header > $@
	sed  's/^/     /'  $< >> $@
	cat txt.header >> $@
	sed  's/^/     /'  $*.txt >> $@

%.html: %.rst
	${RST2HTML} $(<:.rstw=.rst)  > $@

.PHONY: clean
clean:
	rm -f ${RSTs} ${TXTs}
