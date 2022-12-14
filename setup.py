#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
from distutils.sysconfig import get_config_vars

from setuptools import setup
import setuptools

# Bail on Python < 3
assert sys.version_info[0] >= 3

with open("README.md") as readme_file:
    readme = readme_file.read()

version = {}
VERSION_PATH = os.path.join("rascil", "version.py")
with open(VERSION_PATH, "r") as file:
    exec(file.read(), version)

# List of requirements cannot contain lines that start with #
# can neither work with git+https ones, hence we'll remove it
# and add it back below with the correct syntax for setup.py
reqs = [
    line.strip()
    for line in open("requirements.txt").readlines()
    if not line.strip().startswith("#")
    and line.strip() != ""
    and not line.strip().startswith("--extra-index-url")
]

# MF. This is a workaround to be able to build the library with MacOS
if sys.platform == "darwin":
    vars = get_config_vars()
    vars["LDSHARED"] = vars["LDSHARED"].replace("-bundle", "-dynamiclib")
    os.environ["CC"] = "clang"

# NB. These are not really Python extensions (i.e., they do not
# Py_Initialize() and they do define main() ), we are just cheating to
# re-use the setuptools build support.

packages = ["rascil"]
packages_data = [
    i
    for p in packages
    for i in glob.glob(p + "/*/")
    + glob.glob(p + "/*/*/")
    + glob.glob(p + "/*/*/*/")
    + glob.glob(p + "/*/*/*/*/")
    + glob.glob(p + "/*/*/*/*/")
]

setup(
    name="rascil",
    version=version["__version__"],
    python_requires=">=3.7",
    description="Radio Astronomy Simulation, Calibration, and Imaging Library",
    long_description=readme + "\n\n",
    author="See CONTRIBUTORS",
    author_email="realtimcornwell@gmail.com",
    url="https://gitlab.com/ska-telescope/external/rascil",
    license="Apache License Version 2.0",
    zip_safe=False,
    classifiers=[
        "Development Status :: Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    packages=(packages + packages_data),
    install_requires=reqs,
    scripts=["bin/get_rascil_data"],
    test_suite="tests",
    tests_require=["pytest"],
)
