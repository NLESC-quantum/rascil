.. _rascil_functions:

.. toctree::
   :maxdepth: 2

Functions
=========

Create empty visibility data set for observation
------------------------------------------------

* For BlockVisibility: :py:func:`rascil.processing_components.visibility.base.create_blockvisibility`

Read existing Measurement Set
-----------------------------

Casacore must be installed for MS reading and writing:

* List contents of a MeasurementSet: :py:func:`rascil.processing_components.visibility.base.list_ms`
* Creates a list of BlockVisibilities, one per FIELD_ID and DATA_DESC_ID: :py:func:`rascil.processing_components.visibility.base.create_blockvisibility_from_ms`

Visibility weighting and tapering
---------------------------------

* Weighting: :py:func:`rascil.processing_components.imaging.weighting.weight_visibility`
* Gaussian tapering: :py:func:`rascil.processing_components.imaging.weighting.taper_visibility_gaussian`
* Tukey tapering: :py:func:`rascil.processing_components.imaging.weighting.taper_visibility_tukey`

Visibility predict and invert
-----------------------------

* Predict by de-gridding visibilities with Nifty Gridder :py:func:`rascil.processing_components.imaging.ng.predict_ng`
* Invert by gridding visibilities with Nifty Gridder :py:func:`rascil.processing_components.imaging.ng.invert_ng`
* Predict BlockVisibility for Skycomponent :py:func:`rascil.processing_components.imaging.dft.dft_skycomponent_visibility`
* Predict Skycomponent from BlockVisibility :py:func:`rascil.processing_components.imaging.dft.idft_visibility_skycomponent`

Deconvolution
-------------

* Deconvolution :py:func:`rascil.processing_components.image.deconvolution.deconvolve_cube` wraps:

 * Hogbom Clean: :py:func:`rascil.processing_components.arrays.cleaners.hogbom`
 * Hogbom Complex Clean: :py:func:`rascil.processing_components.arrays.cleaners.hogbom_complex`
 * Multi-scale Clean: :py:func:`rascil.processing_components.arrays.cleaners.msclean`
 * Multi-scale multi-frequency Clean: :py:func:`rascil.processing_components.arrays.cleaners.msmfsclean`


* Restore: :py:func:`rascil.processing_components.image.deconvolution.restore_cube`

Calibration
-----------

* Create empty gain table: :py:func:`rascil.processing_components.calibration.operations.create_gaintable_from_blockvisibility`
* Solve for complex gains: :py:func:`rascil.processing_components.calibration.solvers.solve_gaintable`
* Apply complex gains: :py:func:`rascil.processing_components.calibration.operations.apply_gaintable`

Coordinate transforms
---------------------

* Phase rotation: :py:func:`rascil.processing_components.visibility.base.phaserotate_visibility`
* Station/baseline (XYZ <-> UVW): :py:mod:`rascil.processing_components.util.coordinate_support`
* Source (spherical -> tangent plane): :py:mod:`rascil.processing_components.util.coordinate_support`

Image
-----

* Image operations: :py:func:`rascil.processing_components.image.operations`
* Import from FITS: :py:func:`rascil.processing_components.image.operations.import_image_from_fits`
* Export from FITS: :py:func:`rascil.processing_components.image.operations.export_image_to_fits`
* Re-project coordinate system: :py:func:`rascil.processing_components.image.operations.reproject_image`
* Smooth image: :py:func:`rascil.processing_components.image.operations.smooth_image`
* FFT: :py:func:`rascil.processing_components.image.operations.fft_image_to_griddata`
* Remove continuum: :py:func:`rascil.processing_components.image.operations.remove_continuum_image`
* Convert polarisation:

 * From Stokes To Polarisation: :py:func:`rascil.processing_components.image.operations.convert_stokes_to_polimage`
 * From Polarisation to Stokes: :py:func:`rascil.processing_components.image.operations.convert_polimage_to_stokes`


Visibility
----------

* Append/sum/divide/QA: :py:func:`rascil.processing_components.visibility.operations.divide_visibility`
* Remove continuum: :py:func:`rascil.processing_components.visibility.operations.remove_continuum_blockvisibility`
* Integrate across channels: :py:func:`rascil.processing_components.visibility.operations.integrate_visibility_by_channel`


