.. _documentation_master:

.. toctree::

Radio Astronomy Simulation, Calibration and Imaging Library
###########################################################

The Radio Astronomy Simulation, Calibration and Imaging Library expresses radio interferometry calibration and
imaging algorithms in python and numpy. The interfaces all operate with familiar data structures such as image,
visibility table, gain table, etc. The python source code is directly accessible from these documentation pages:
see the source link in the top right corner.

As of version 0.2.0, the data classes are built on the `Xarray <https:/xarray.pydata.org>`_ library, offering a
rich API for applications. For more details including how to update existing scripts, see
:ref:`Use of xarray <RASCIL_xarray>`.

To achieve sufficient performance we take a dual pronged approach - using threaded libraries for shared memory
processing, and the `Dask <https:/www.dask.org>`_ library for distributed processing.


The role of the RASCIL in SKA Science Data Processing (SDP)
===========================================================

RASCIL was developed in SDP under the name ARL (Algorithm Reference Library) with the emphasis of creating reference
versions of standard algorithms. The ARL was therefore designed to present primarily imaging algorithms in a simple
Python-based form so that the implemented functions could be seen and understood easily. This also fulfilled the
requirement of providing a simple test version where algorithms could be tested and compared as necessary.

For an overview of the SDP see the `SDP CDR
documentation <http://ska-sdp.org/publications/sdp-cdr-closeout-documentation>`_

More details can be found at: `SKA1 SDP Algorithm Reference Library (ARL) Report <http://ska-sdp.org/sites/default/files/attachments/ska-tel-sdp-0000150_02_sdparlreport_part_1_-_signed.pdf>`_

Subsequent to the conclusion of the SDP project, it became clear that ARL could play a larger role than being limited
to a reference library. Hence, it was renamed to the Radio Astronomy Simulation, Calibration and Imaging Library
(RASCIL) and is undergoing continued development. The Algorithm Reference Library (ARL) is now frozen. The background
motivation and requirements of the ARL/RASCIL are detailed further in :ref:`Background <RASCIL_background>`.


.. toctree::
   :maxdepth: 2

   RASCIL_install
   RASCIL_examples
   RASCIL_structure
   RASCIL_api
   RASCIL_development

* :ref:`genindex`
* :ref:`modindex`

.. _feedback: mailto:realtimcornwell@gmail.com
