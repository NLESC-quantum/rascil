.. _rascil_wagg:

RASCIL and WAGG
***************

RASCIL can use GPU-based version of nifty-gridder called WAGG for the gridding-degridding operations:

https://gitlab.com/ska-telescope/sdp/ska-gridder-nifty-cuda/-/tree/sim-874-python-wrapper

The are two function counterparts to predict_ng and invert_ng called predict_wg and invert_wg,
:py:func:`rascil.processing_components.imaging.wg.invert_wg`
:py:func:`rascil.processing_components.imaging.wg.predict_wg`

In order to use WAGG it is required to install it separately from it's repository after RASCIL installation. WAGG uses numpy to build the installation wheel,
and it will download the recent one if numpy is absent in a system. The numpy version mismatch can cause the WAGG crash.  

Installing WAGG module
==================================

To install WAGG it is required to clone the repository, switch to the python wrapper branch, change to `python` folder and run `pip install .` , e.g.

* git clone https://gitlab.com/ska-telescope/sdp/ska-gridder-nifty-cuda.git
* cd ska-gridder-nifty-cuda
* git fetch
* git checkout --track origin/sim-874-python-wrapper
* cd python
* pip install .

Alternatively, WAGG can be install directly by single `pip` command,

* pip install git+http://gitlab.com/ska-telescope/sdp/ska-gridder-nifty-cuda.git@sim-874-python-wrapper#subdirectory=python


