.. _rascil_macos_install:

Installation of RASCIL on macos
===============================

RASCIL is well-suited to running under macos. Installation should be straightforward. Although the pip approach can
be used, we recommend use of Anaconda https://www.anaconda.com. It is necessary to
install python-casacore in a separate step. The steps required are::

    conda create -n rascil_env python=3.7
    conda activate rascil_env
    conda install -c conda-forge python-casacore=3.3.1
    git clone https://gitlab.com/ska-telescope/rascil
    cd rascil
    pip install -r requirements.txt

Then at the top level directory, do::

    pip install -e .

This will install it as a development package (this adds it into the path in situ).

Finally to get the casa measures data::

    rsync -avz rsync://casa-rsync.nrao.edu/casa-data/geodetic /opt/anaconda/envs/rascil/lib/casa/data/

Or if your anaconda is in your home directory::

    rsync -avz rsync://casa-rsync.nrao.edu/casa-data/geodetic ~/opt/anaconda/envs/rascil/lib/casa/data/


Finally, put the following definitions in your .bashrc::

    export RASCIL=/path/to/rascil
    export PYTHONPATH=$RASCIL:$PYTHONPATH


.. _feedback: mailto:realtimcornwell@gmail.com
