0.2.2b0
-------

* [MR207] RCAL prototype as an app
* [MR206] Add AA0.5 definitions to configurations
* [MR203] Updated the CI pipeline to build the docker images, build on tag, and publish to the Central Artefact Repository
* [MR201] Calculation of CLEAN threshold should avoid uncleaned regions
* [MR200] Migrated docker builds from rascil-docker repository to rascil/docker directory
* [MR199] Added CLI option and env variable that passes dask scheduler file to setup dask cluster
* [MR198] Fixes termination problem in CLEAN caused by error in threshold determination
* [MR191] Updated imaging_qa to make it optional to plot diagnostics plots
* [MR189] CI pipeline update: tests run in two separate jobs: dask/non-dask
* [MR185] Updated function to generate MID simulation skycomponents
* [MR183, MR195] Update CI pipeline to publish Python package in the SKA central artefact repository
* [MR181] Large set of changes to reduce run time of CI tests from about 160 minutes to 50 minutes
* [MR180] RA, Dec coordinates now added to Image when needed, saves time and memory
* [MR177] RFI simulation code is refactored to be compatible with the standard RFI input data
* [MR174] Improves the memory and processing scaling of the Multi-Scale Multifrequency Clean
* [MR169] Added functionality in imaging_qa to analyse frequency moment images to allow more accurate calculation of spectral index
* [MR161, MR163] Renamed ci_checker to imaging_qa
* [MR160] Changes aimed at decreasing the memory used in the restore step.
* [MR154] The repository is Black-compatible, and expects all python code to be [Black](https://github.com/psf/black) formatted
* [MR152] Tools for measuring time spent in Dask functions
* [MR150] Updated ci_checker to provide functionality for primary beam correction
* [MR148] Initialise astropy before dask starts, minimises threading errors
* [MR142] Updated ci_checker to analyse multi frequency FITS images
* [MR139] Simple app rascil_image_check to check value of image, to be used in scripts
* [MR138] Added wide field plots for ci_checker
* [MR136] Added option to load ci_checker arguments in the command line directly from a file
* [MR132, MR145, MR158] ci_checker now produces various diagnostics plots, e.g. power spectrum, running mean, histogram, spectral index with flux cutoff, etc.
* [MR131] Supports modelling of compact sources at fractional pixel locations in the continuum imaging pipelines. 
* [MR130, MR133, MR141] Updated ci_checker's logging and errors, as well as rearranging the files' locations etc. 
* [MR123, MR125] Added index files to the ci_checker, which list and point to the output files of the app
* [MR124] Clean beam now propagated through inputs to rascil-imager and out to FITS 
* [MR122] Added functions to plot sources in skycomponents format compatible for ci_checker

0.2.1
-----
* New app rascil_advise to advise on imaging parameters for a MeasurementSet
* New app ci_checker to check sources in continuum imaging images with BDSF
* Implement using requirements*.in files for dependency management.
  Add files: requirements.in, requirements-docs.in, requirements-test.in
* Makefile updated with requirements management commands:
  `make requirements`, `make install_requirements`, `make update_requirements`
* Added a new job, called `compile_requirements`, to the CI pipeline to update the requirements;
  it is triggered by a schedule in GitLab
* There is a new option to read in array configurations file with long,lat and specify an arbitrary height.

0.2.0
-----

* This is a major upgrade breaking interfaces. To bring a wide range of improved capabilities,
  we have switched to use xarray instead of numpy to hold class
  data. Thus for example myimage.data now is an xarray.DataArray, and myblockvis.data is
  an xarray.DataSet. By appending .values, you can get the usual numpy arrays. Xarray has many
  capabilities that we intend to make use of in the future. The simplest description is that Xarray
  provides named coordinates such as time, frequency, l, m, baseline.
* The Image class now is xarray.DataArray with axes "l", "m", "polarisation", "frequency"
* There is now only one holder of visibilities: BlockVisibility. This is now baseline- rather than
  antenna-oriented. The axes are "time", "baselines", "frequency", "polarisation", "uvw_index".
* All data classes now support xarray operators such as sel, isel, groupby, groupby_bins,
  where. The selections can be specified as dictionaries
  holding slices e.g. {"time":slice("2001-01-02T09:00:00.0", "2001-01-02T10:00:00.0")}
* To improve performance, all imaging algorithms have now moved from the workflows layer to be done in
  processing_components. The nifty gridder is automatically installed and used as the default algorithm,
  though 2d and awprojection are also available. predict_list_rsexecute_workflow and invert_list_rsexecute_workflow now simply
  distribute processing across a list of BlockVisibility's. The available contexts are 2d, ng, and
  awprojection.
* Other workflows such as the pipelines work as before.
* The LOW test beam is now a uniformly weighted 38m diameter disk
* There is a new category of python code: command line apps. The first of these is the
  rascil/apps/rascil-imager.sh
* Array configurations are now specified only in global coordinates: XYZ or long, lat, height.
* We have chosen to drop installation via a Conda environment file. The instructions have been
  updated.

0.1.9
-----

* Documentation structure and content improved after review by othe SKA teams
* Add MeerKAT+ configuration and MEERKAT_B2 voltage patterns
* predict, restore of SkyModel now possible
* Multiple changes to support ska-mid-simulations
* Error in uvw calculations introduced and corrected. UVW calculations require
configurations in local coordiates

0.1.8
-----

* Changes due to move to SKA repository: pip and docker files now available from nexus.engageska-portugal.pt
* Pip file now downloads all prerequisites
* Unit tests reduced in scale to improve turnaround of Gitlab tests

0.1.7
------

 * Docker files split to separate repo: ska-telescope/rascil-docker
 * Build of Docker images triggered by successful build of rascil/master
 * Size of data reduced by preselecting columns in GLEAM
 * Now using casacore.measures for geometry calculations
 * Removed round trip errors in reading/writing/reading MeasurementSets
 * Channel averaging on read of MeasurementSets
 * Robust weighting now added
 * Nifty gridder supported (and available in docker images)
 * WTowers gridder supported (requires separate installation)
