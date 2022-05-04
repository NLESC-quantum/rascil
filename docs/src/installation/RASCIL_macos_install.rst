.. _rascil_macos_install:

Installation of RASCIL on macos
===============================

RASCIL is well-suited to running under macos. Installation should be straightforward. Although the pip approach can
be used, we recommend use of Anaconda https://www.anaconda.com It is necessary to
install python-casacore in a separate step. The steps required are::

    conda create -n rascil_env python=3.9
    conda activate rascil_env
    conda install -c conda-forge python-casacore
    git clone https://gitlab.com/ska-telescope/external/rascil.git
    cd rascil
    make install_requirements

We have specified python 3.9 since this currently is the preferred and supported version.

Finally, put the following definitions in your .bashrc::

    export RASCIL=/path/to/rascil
    export PYTHONPATH=$RASCIL:$PYTHONPATH


.. _feedback: mailto:realtimcornwell@gmail.com
