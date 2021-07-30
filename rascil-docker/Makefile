pull_all:
	cd rascil-base; make pull;
	cd rascil-full; make pull;
	cd rascil-notebook; make pull;
	cd rascil-imaging-qa; make pull;

build_all:
	cd rascil-base; make build;
	cd rascil-full; make build;
	cd rascil-notebook; make build;
	cd rascil-imaging-qa; make build;

build_push_all:
	cd rascil-base; make build; make tag_latest; make push_latest
	cd rascil-full; make build; make tag_latest; make push_latest
	cd rascil-notebook; make build; make tag_latest; make push_latest
	cd rascil-imaging-qa; make build; make tag_latest; make push_latest

push_all:
	cd rascil-base; make tag_latest; make push_latest
	cd rascil-full; make tag_latest; make push_latest
	cd rascil-notebook; make tag_latest; make push_latest
	cd rascil-imaging-qa; make tag_latest; make push_latest

push_all_stable:
	cd rascil-base; make tag_stable; make push_stable
	cd rascil-full; make tag_stable; make push_stable
	cd rascil-notebook; make tag_stable; make push_stable
	cd rascil-imaging-qa; make tag_stable; make push_stable

push_all_release:
	cd rascil-base; push_version
	cd rascil-full; push_version
	cd rascil-notebook; push_version
	cd rascil-imaging-qa; push_version

rm_all:
	cd rascil-base; make rm
	cd rascil-full; make rm
	cd rascil-notebook; make rm
	cd rascil-imaging-qa; make rm

ls_all:
	cd rascil-base; make ls
	cd rascil-full; make ls
	cd rascil-notebook; make ls
	cd rascil-imaging-qa; make ls

WORKDIR=./test_results

IMAGES=artefact.skao.int

# Docker tests
test_base:
	- echo "Running Dask-enabled test cluster_test_ritoy - takes about 20 -- 30 seconds"
	echo "cd ${WORKDIR};python3 /rascil/examples/cluster_tests/ritoy/cluster_test_ritoy.py" | docker run -i  \
	${IMAGES}/rascil-base /bin/bash -e

test_full:
	- echo "Running imaging.py - takes about 20 -- 30 seconds, writes three fits files in ${WORKDIR}"
	echo "cd ${WORKDIR};python3 /rascil/examples/scripts/imaging.py" | docker run -i \
	${IMAGES}/rascil-full /bin/bash -e

test_notebook:
	- echo "Running imaging notebook - takes about 20 -- 30 seconds, writes html file ${WORKDIR}/imaging.html"
	echo "cd ${WORKDIR};cp /rascil/examples/notebooks/imaging.ipynb .;jupyter nbconvert --execute --to html \
	--ExecutePreprocessor.timeout=1200 imaging.ipynb" | docker run -i ${IMAGES}/rascil-notebook /bin/bash -e

test_imaging_qa:
	- echo "Running imaging QA code - takes about 3-5 seconds, outputs the help of the QA tool"
	docker run -i ${IMAGES}/rascil-imaging-qa /bin/bash -e

# Singularity tests
test_base_singularity:
	- echo "Running Dask-enabled test cluster_test_ritoy - takes about 20 -- 30 seconds"
	- singularity pull RASCIL-base.img docker://${IMAGES}/rascil-base
	singularity exec RASCIL-base.img "cd ${WORKDIR};python3 /rascil/examples/cluster_tests/ritoy/cluster_test_ritoy.py"

test_full_singularity:
	- echo "Running imaging.py - takes about 20 -- 30 seconds, writes three fits files in ${WORKDIR}"
	- singularity pull RASCIL-full.img docker://${IMAGES}/rascil-full
	singularity exec RASCIL-full.img "cd ${WORKDIR};python3 /rascil/examples/scripts/imaging.py"

test_notebook_singularity:
	- echo "Running imaging notebook - takes about 20 -- 30 seconds, writes html file ${WORKDIR}/imaging.html"
	- singularity pull RASCIL-notebook.img docker://${IMAGES}/rascil-notebook
	singularity exec RASCIL-notebook.img "cd ${WORKDIR};jupyter nbconvert --execute \
	--ExecutePreprocessor.timeout=1200 --to html /rascil/examples/notebooks/imaging.ipynb"

test_imaging_qa_singularity:
	- echo "Running imaging QA code - takes about 3-5 seconds, outputs the help of the QA tool"
	- singularity pull rascil-imaging-qa.img docker://${IMAGES}/rascil-imaging-qa
	singularity run rascil-imaging-qa.img

lint:
# outputs linting.xml
	pylint --exit-zero --output-format=pylint2junit.JunitReporter rascil-base > linting.xml
	pylint --exit-zero --output-format=parseable rascil-base
	black --check .

test_unittest:
	HOME=`pwd` py.test \
        tests --verbose \
	--junitxml unit-tests.xml \
	--cov rascil-base \
	--cov-report term \
	--cov-report html:coverage  \
	--cov-report xml:coverage.xml

test_all: test_base test_full test_notebook test_imaging_qa

test_all_singularity: test_base_singularity test_full_singularity test_notebook_singularity test_imaging_qa_singularity

everything: rm_all build_all test_all

.phony: build_all rm_all test_all test_base test_full test_notebook test_imaging_qa \
    test_all_singularity test_base_singularity test_full_singularity test_notebook_singularity \
