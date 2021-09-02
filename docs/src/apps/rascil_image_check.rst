.. _rascil_apps_rascil_image_check:

.. py:currentmodule:: rascil.apps

.. toctree::
   :maxdepth: 3

==================
rascil_image_check
==================

rascil_image_check is a command line app written using RASCIL. It allows simple
check on an image statistics.

The allowed fields are the statistics checked by :py:func:`rascil.processing_components.image.operations.qa_image`

Example script
++++++++++++++

The following provides a check on the maximum of an image suitable for use in a shell script.
The value returned is 0 if the constraint is obeyed and 1 if not::

    python3 $RASCIL/rascil/apps/rascil_image_check.py --image $RASCIL/data/models/M31_canonical.model.fits --stat max --min 0.0 --max 1.2

Command line arguments
++++++++++++++++++++++

.. argparse::
   :filename: ../../rascil/apps/rascil_image_check.py
   :func: cli_parser
   :prog: rascil_image_check.py
