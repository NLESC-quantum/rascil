.. _rascil_install:

Installation
============

RASCIL can be run on a Linux or macOS machine or cluster of machines, using python 3.7 or 3.8. At least 16GB physical
memory is necessary to run the full test suite. In general more memory is better. RASCIL uses Dask for
multi-processing and can make good use of multi-core and multi-node machines.

Installation via pip
++++++++++++++++++++

If you just wish to run the package and do not intend to run simulations or tests, RASCIL can be installed using pip::

     pip3 install --index-url=https://artefact.skao.int/repository/pypi-all/simple rascil

This will download the latest stable version.

This will download and install the python files in the rascil, and dependencies. For simulations, you must add the data
in a separate step::

    mkdir rascil_data
    cd rascil_data
    curl https://ska-telescope.gitlab.io/external/rascil/rascil_data.tgz -o rascil_data.tgz
    tar zxf rascil_data.tgz
    cd data
    export RASCIL_DATA=`pwd`

If you wish to run the RASCIL examples or tests, use one of the steps below.

Installation via docker
+++++++++++++++++++++++

If you are familiar with docker, an easy approach is to use docker:

 .. toctree::
    :maxdepth: 1

    installation/RASCIL_docker


Installation via git clone
++++++++++++++++++++++++++

Use of git clone is necessary if you wish to develop and possibly contribute RASCIL code. Installation should be straightforward. We strongly recommend the use of a python virtual environment.

RASCIL requires python 3.7 or 3.8.

The installation steps are:

- Use git to make a local clone of the Github repository::

   git clone https://gitlab.com/ska-telescope/external/rascil.git

- Change into that directory::

   cd rascil

- Use pip to install required python packages::

   pip3 install pip --upgrade
   pip3 install -r requirements.txt

- Setup RASCIL::

   python3 setup.py install

- RASCIL makes use of a number of data files. These can be downloaded using Git LFS::

    pip3 install git-lfs
    git lfs install
    git-lfs pull

If git-lfs is not already available, then lfs will not be recognised as a valid option for git in the second step.
In this case, git-lfs can be installed via :code:`sudo apt install git-lfs` or from a `tar file <https://docs.github.com/en/github/managing-large-files/installing-git-large-file-storage>`_

- Put the following definitions in your .bashrc::

    export RASCIL=/path/to/rascil
    export PYTHONPATH=$RASCIL:$PYTHONPATH

:code:`python setup.py install` installs in the correct site-packages location so the definition of PYTHONPATH is not needed
if you only don't intend to update or edit rascil in place. If you do intend to make changes, you will need the
definition of PYTHONPATH.

CASA measures data
++++++++++++++++++

We use casacore for some coordinate conversions. As a result the CASA measures data files are needed. Depending on
your setup, these may already be in the right place. If not, you can download a copy and tell casacore where it is::

    echo 'measures.directory: ~/casacore_data' > ~/.casarc
    rsync -avz rsync://casa-rsync.nrao.edu/casa-data/geodetic ~/casacore_data

If you get errors about the UTC table being out of date, typically of the form::

    2020-08-07 15:37:59 SEVERE MeasTable:...+ Leap second table TAI_UTC seems out-of-date.
    2020-08-07 15:37:59 SEVERE MeasTable:...+ Until the table is updated (see the CASA documentation or your system admin),
    2020-08-07 15:37:59 SEVERE MeasTable:...+ times and coordinates derived from UTC could be wrong by 1s or more.

you should repeat the rsync.

Installation on specific machines
+++++++++++++++++++++++++++++++++

.. toctree::
   :maxdepth: 2

   installation/RASCIL_macos_install
   installation/RASCIL_CSD3_install
   installation/RASCIL_galahad_install
   installation/RASCIL_P3_install

Trouble-shooting
++++++++++++++++

Testing
^^^^^^^

Check your installation by running a subset of the tests::

   pip3 install pytest pytest-xdist
   py.test -n 4 tests/processing_components

Or the full set::

   py.test -n 4 tests

- Ensure that pip is up-to-date. If not, some strange install errors may occur.
- Check that the contents of the data directories have plausible contents. If gif-lfs has not been run successfully then the data files will just contain meta data, leading to strange run-time errors.
- There may be some dependencies that require either conda (or brew install on a mac).
- Ensure that you have made the directory test_results to store the test results. Or add the below in the test script::
  if not os.path.isdir(test_results):os.makedirs(test_results)

Casacore installation
^^^^^^^^^^^^^^^^^^^^^

RASCIL requires python-casacore to be installed. This is included in the requirements for the RASCIL install and so
should be installed automatically via pip. In some cases there may not be a compatible binary install (wheel) available
via pip. If not, pip will download the source code of casacore and attempt a build from source. The most common failure
mode during the source build is that it cannot find the boost-python libraries. These can be installed via pip. If
errors like this occur, once rectified, re-installing python-casacore separately via pip may be required, prior to
re-commencing the RASCIL install.
Trouble-shooting problems with a source install can be difficult. If available, this can be avoided by using anaconda as
the base for an environment. It supports python-casacore which can be installed with “conda install python-casacore”.
It may also be possible to avoid some of the more difficult issues with building python-casacore by downloading CASA
prior to the RASCIL install.

RASCIL data in notebooks
^^^^^^^^^^^^^^^^^^^^^^^^

In some case the notebooks may not automatically find the RASCIL data directory, in which case explicitly setting the
RASCIL_DATA environment variable may be required: :code:`%env RASCIL_DATA=~/rascil_data/data`.

.. _feedback: mailto:realtimcornwell@gmail.com
