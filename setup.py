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

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as reqs_file:
    reqs = reqs_file.read()

# MF. This is a workaround to be able to build the library with MacOS
if sys.platform == 'darwin':
    vars = get_config_vars()
    vars['LDSHARED'] = vars['LDSHARED'].replace('-bundle', '-dynamiclib')
    os.environ["CC"] = "clang"

# NB. These are not really Python extensions (i.e., they do not
# Py_Initialize() and they do define main() ), we are just cheating to
# re-use the setuptools build support.

setup(name='rascil',
      version='0.1.8b0',
      python_requires='>=3.6',
      description='Radio Astronomy Simulation, Calibration, and Imaging Library',
      long_description=readme + '\n\n',
      author='Tim Cornwell, Peter Wortmann, Bojan Nikolic, Feng Wang, Vlad Stolyarov, Danielle Fenech, Mark Ashdown, Danielle Fenech',
      author_email='realtimcornwell@gmail.com',
      url='https://gitlab.com/ska-telescope/rascil',
      license='Apache License Version 2.0',
      zip_safe=False,
      classifiers=[
          'Development Status :: Alpha',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: BSD License',
          'Natural Language :: English',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7'],
      packages=setuptools.find_namespace_packages(where="rascil"),
      package_dir={"": "rascil"},
      test_suite='tests',
      install_requires=reqs,
      setup_requires=[
          # dependency for `python setup.py test`
          'pytest-runner',
          # dependencies for `python setup.py build_sphinx`
          'sphinx',
          'recommonmark'],
      tests_require=[
          'pytest',
          'pytest-cov',
          'pytest-json-report',
          'pytest-xdist',
          'pycodestyle']
      )
