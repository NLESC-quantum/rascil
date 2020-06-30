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
