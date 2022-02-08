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
======================

To install WAGG it is required to clone the repository, switch to the python wrapper branch, change to `python` folder and run `pip install .` , e.g.

* git clone https://gitlab.com/ska-telescope/sdp/ska-gridder-nifty-cuda.git
* cd ska-gridder-nifty-cuda
* git fetch
* git checkout --track origin/sim-874-python-wrapper
* cd python
* pip install .

Alternatively, WAGG can be install directly by single `pip` command,

* pip install git+http://gitlab.com/ska-telescope/sdp/ska-gridder-nifty-cuda.git@sim-874-python-wrapper#subdirectory=python

Using WAGG GPU-based predict and invert functions
=================================================

WAGG module makes a use of Nvidia runtime system, called `NVRTC`. It is a runtime compilation library for CUDA C++. 
It accepts CUDA C++ source code in character string form and creates handles that can be used to obtain the PTX.
WAGG module can be built without runtime libraries installed, but they should be installed in a system before
using WAGG. In a Linux system the libraries are usually `lib64/libnvrtc*.so` . More information on `NVRTC` can be found on CUDA website,
https://docs.nvidia.com/cuda/nvrtc/index.html .

When the runtime support is installed, the functions predict_wg and invert_wg can be used as the CPU-based predict_ng and invert_ng since the parameters are the same.
