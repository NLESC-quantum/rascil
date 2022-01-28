master
------
* Add ao.py which directly copied a few AOTOOL functions ([MR306](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/306))
* Remove Numba related code ([MR303](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/303))
* Install PyBDSF from PyPi test_skymodel_extract_skycomponentsusing latest python3.9 wheel ([MR300](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/300))
* Bug fix in initial SkyModel for ICAL: convert polarisation of SkyComponents read from file to image polarisation ([MR296](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/296))
* Add uv and r visibility selection in rascil_imager ([MR294](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/294))

0.5.0
-----

* Add DASK support to Quality Assesment app ([MR286](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/286))
* Added blockvisibility selection documentation and functions ([MR292](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/292))
* Developed the capability to apply realistic LOW beam for skymodel in the RCAL app and refined its application in imaging_qa ([MR284](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/284))
* The rascil_imager option for global solution was being ignored ([MR278](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/278))
* The numba version of the DFT became unstable and is no longer part of the test suite ([MR278](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/278))
* Solutions for Jones type T and G now integrate across BlockVisibility spectrum ([MR278](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/278))
* Gaintables now labelled by Jones type (e.g. T, G, or B) ([MR278](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/278))
* Reset of skymodel in ICAL now sets both image and components ([MR278](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/278))
* create_blockvisibility can accepts time in UTC as well as HA (both in radians) ([MR278](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/278))
* Pipelines produce more log info on calibration progress, also writes gaintables ([MR278](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/278))
* Prototype of an app to calculate sensitivity for weightings and tapers ([MR273](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/273))
* Update the RCAL pipeline to use RFI flags ([MR288](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/288), [MR280](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/280))
* Added option to use and generate initial SkyModel in ICAL ([MR274](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/274))
* Added a CI job to build and test the docker images upon every merge to master ([MR270](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/270))
* Fixed an issue in plot_skycomponent to deal with the angle wrap problem([MR269](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/269))
* Large number of test fixes and improvements ([MR267](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/267))
* create_blockvisibility_iterator removed in favor of xarray.groupby ([MR267](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/267))
* skymodel_predict_calibrate and skymodel_calibrate_image functions added ([MR267](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/267))
* Two dimensional transforms done with nifty gridder, do_wstacking=False ([MR267](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/267))
* Model Partition Calibration Proof-Of-Concept code removed ([MR267](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/267))
* Shared, serial workflow types removed, functionality moved to processing_components ([MR267](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/267))
* Restructuring of invert, predict to invert_blockvisibility, predict_blockvisibility, context added ([MR267](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/267))

0.4.0
-----

* Fixed bug in hogbom clean ([MR265](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/265))
* imaging-pipelines notebook now larger, runs in a few minutes.([MR258](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/258))
* The pipelines now consume less memory when running.([MR254](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/254))
* All image-based pipelines have been removed in favour of skymodel-based pipelines ([MR250](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/250))
* fit_psf was incorrectly converting stddev to FWHM, resulting in beams about 30% too big. ([MR246](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/246))
* The ska-pipelines examples have been improved ([MR247](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/247))
* Fixed issue within flux plots in plot_skycomponent for multiple polarisations ([MR263](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/263))
* Updated imaging_qa to improve the spectral index calculation ([MR261](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/261))
* Persist dask graph at the end of each major cycle and add sizeof method to dask data objects ([MR249](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/249))
* Replaced running RASCIL-as-a-cluster via docker-compose with kubernetes/minikube ([MR238](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/238), [MR244](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/244))
* Updated RFI code to standardise Low beam gain inputs ([MR229](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/229))
* Use astropy instead of casacore for geometry ([MR227](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/227))
* Fixed angle wrap problem in plot_skycomponents ([MR225](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/225))
* Updated documentation on installation to reflect recent changes ([MR223](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/223), [MR232](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/232))
* Added Mid beam calculations to RFI code ([MR204](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/204))
* RASCIL is running on python3.9 (docker images and CI pipelines are also updated) ([MR219](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/219))
* Further development of RCAL app including the gaintable plot functions ([MR212](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/212))
* Fixed docker make commands and update documentation with new skao links ([MR218](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/218))
* Added script to download RASCIL and casacore data ([MR215](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/215))
* Removed unused data from Docker images, removed redundant steps and fixed dependency checking ([MR251](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/251))

0.3.0
-----

* RCAL prototype as an app ([MR207](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/207))
* Add AA0.5 definitions to configurations ([MR206](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/206))
* Updated the CI pipeline to build the docker images, build on tag, and publish to the Central Artefact Repository ([MR203](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/203))
* Calculation of CLEAN threshold should avoid uncleaned regions ([MR201](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/201))
* Migrated docker builds from rascil-docker repository to rascil/docker directory ([MR200](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/200))
* Added CLI option and env variable that passes dask scheduler file to setup dask cluster ([MR199](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/199))
* Fixes termination problem in CLEAN caused by error in threshold determination ([MR198](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/198))
* Updated imaging_qa to make it optional to plot diagnostics plots ([MR191](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/191))
* CI pipeline update: tests run in two separate jobs: dask/non-dask ([MR189](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/198))
* Updated `create_mid_simulation_components` function to generate MID simulation skycomponents ([MR185](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/185))
* Update CI pipeline to publish Python package in the SKA central artefact repository ([MR183](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/183), [MR195](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/195))
* Large set of changes to reduce run time of CI tests from about 160 minutes to 50 minutes ([MR181](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/181))
* RA, Dec coordinates now added to Image when needed, saves time and memory ([MR180](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/180))
* RFI simulation code is refactored to be compatible with the standard RFI input data ([MR177](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/177))
* Improves the memory and processing scaling of the Multi-Scale Multifrequency Clean ([MR174](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/174))
* Added functionality in imaging_qa to analyse frequency moment images to allow more accurate calculation of spectral index ([MR169](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/169))
* Renamed ci_checker to imaging_qa ([MR161](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/161), [MR163](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/163))
* Changes aimed at decreasing the memory used in the restore step. ([MR160](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/160))
* The repository is Black-compatible, and expects all python code to be [Black](https://github.com/psf/black) formatted ([MR154](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/154))
* Tools for measuring time spent in Dask functions ([MR152](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/152))
* Updated ci_checker to provide functionality for primary beam correction ([MR150](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/150))
* Initialise astropy before dask starts, minimises threading errors ([MR148](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/148))
* Updated ci_checker to analyse multi frequency FITS images ([MR142](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/142))
* Simple app rascil_image_check to check value of image, to be used in scripts ([MR139](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/139))
* Added wide field plots for ci_checker ([MR138](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/138))
* Added option to load ci_checker arguments in the command line directly from a file ([MR136](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/136))
* ci_checker now produces various diagnostics plots, e.g. power spectrum, running mean, histogram, spectral index with flux cutoff, etc. ([MR132](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/132), [MR145](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/145), [MR158](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/158))
* Supports modelling of compact sources at fractional pixel locations in the continuum imaging pipelines ([MR131](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/131))
* Updated ci_checker's logging and errors, as well as rearranging the files' locations etc. ([MR130](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/130), [MR133](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/133), [MR141](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/141))
* Replaced nifty-gridder with the [ducc0](https://pypi.org/project/ducc0/) version ([MR129](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/129))
* Added index files to the ci_checker, which list and point to the output files of the app ([MR123](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/123), [MR125](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/125))
* Clean beam now propagated through inputs to rascil-imager and out to FITS ([MR124](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/124)) 
* Added functions to plot sources in skycomponents format compatible for ci_checker ([MR122](https://gitlab.com/ska-telescope/external/rascil/-/merge_requests/122))

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
