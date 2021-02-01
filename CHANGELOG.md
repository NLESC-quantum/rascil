0.2.1b0
-------
* Implement using requirements*.in files for dependency management.
  Add files: requirements.in, requirements-docs.in, requirements-test.in
* Makefile updated with requirements management commands:
  `make requirements`, `make install_requirements`, `make update_requirements`
* Added a new job, called `compile_requirements`, to the CI pipeline to update the requirements; 
  it is triggered by a schedule in GitLab

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
