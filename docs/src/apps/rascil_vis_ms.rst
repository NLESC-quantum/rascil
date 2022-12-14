.. _rascil_apps_rascil_vis_ms:

.. py:currentmodule:: rascil.apps

.. toctree::
   :maxdepth: 3

=============
rascil_vis_ms
=============

rascil_vis_ms is a command line app written using RASCIL for simple visualisation of an MS. It's primary use is
for the RFI simulations.

Example script
++++++++++++++

The following runs the visualisation on an MS generated by the RFI simulations::

    #!/bin/bash
    # Run this in the directory containing ./simulate_rfi.ms
    python3 $RASCIL/rascil/apps/rascil_vis_ms.py --ingest_msname ./simulate_rfi.ms

Command line arguments
++++++++++++++++++++++

.. argparse::
   :filename: ../../rascil/apps/rascil_vis_ms.py
   :func: cli_parser
   :prog: rascil_vis_ms.py
