#!/bin/bash

# =========================================================== #
# Set of comands to Install rascil and wrapper into Linux system #
# =========================================================== #


# If dependency environment modules are loaded and repository is already cloned
# this file can be sources from RASCIL and should build the system
# i.e. rascil> source scripts/build-rascil-linux-new.sh


# Load the dependency modules
# PYTHON 3.6 +
# GIT 2.10+
# GIT-LFS 
# i.e.:
#module load python-3.5.2-gcc-5.4.0-rdp6q5l
#module load python-3.6.1-gcc-5.4.0-23fr5u4
#module load git-2.14.1-gcc-5.4.0-acb553e
#module load git-lfs-2.3.0-gcc-5.4.0-oktvmkw
#module load cfitsio-3.410-gcc-5.4.0-tp3pkyv

# Clone the repository
#git clone https://github.com/SKA-ScienceDataProcessor/rascil/
#cd rascil/

# Get the data
#git-lfs pull

# Start the building ARL through building a python virtualenvironment
virtualenv -p `which python3` _build
source _build/bin/activate
pip install --upgrade pip
pip install -U setuptools
pip install coverage numpy
pip install -r requirements.txt 
pip install virtualenvwrapper
source virtualenvwrapper.sh

# Build the ARL and C Wrappers (FFI) if possible
python setup.py install

# Add ARL to the PATH
# An alternative if not using virtualenv is to:
# export PYTHONPATH=$PWD:$PWD/ffiwrappers/src:$PYTHONPATH
add2virtualenv $PWD
add2virtualenv $PWD/ffiwrappers/src

# Setup RASCIL
export RASCIL=$PWD

# Test it
cd workflows/ffiwrapped/serial
make run

