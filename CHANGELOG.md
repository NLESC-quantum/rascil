
0.2.0b0
-------

* Major upgrade breaking interfaces. Switched to use xarray instead of numpy to hold class data
* LOW test beam is now a uniformly weighted 38m diameter disk

0.1.10b0
--------


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
