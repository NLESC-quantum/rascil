.. _manage_requirements:

Managing requirements
*********************

RASCIL requirements are stored in three files:

 * ``requirements.in`` Python requirements for the main code base
 * ``requirements-test.in`` Python requirements to run the tests
 * ``requirements-docs.in`` Python requirements to build the documentation

``pip-compile`` is used to generate the corresponding .txt files. ``pip-compile`` resolves
all dependencies and saves them with their resolved versions in the .txt files.

This method is used to make sure we do not update requirements with every build,
but rather install them from the .txt files, where they are pinned. We also have to
make sure we regularly update these versions, by running ``pip-compile`` on the
.in files, which ideally do not contain version pins.

Manually updating the requirements
----------------------------------

The ``Makefile`` of RASCIL contains three options to work with requirements
on your local machine:

 * :code:`make requirements` This will update the requirements in the .txt file, but will not install them
 * :code:`make install_requirements` This will install the existing requirements from the .txt files, but not update them
 * :code:`make update_requirements` This will first update all requirements, then install them (i.e it runs the first two commands)

The first and third commands change the .txt files, but do not commit the changes.
Still, it is worth running them from a branch, and not directly from master.

Process automation
------------------

Regularly updating the requirements manually is prone to be forgotten, which
can result in packages being out-of-date very quickly. Hence we set up a semi-automatic
process using the GitLab CI pipeline with a job run on a schedule.

The scheduled pipeline only runs one job, with the following steps:

 * run :code:`make requirements`
 * check if there are changes compared to the existing remote files
 * if there, create and check out a new branch
 * commit and push the changes to the new branch
 * create a Merge Request (MR) of the new branch into the source branch
 * assign the MR
 * if there aren't any changes, do nothing

The tests are not run as part of this pipeline, because the MR created
at the end of will have the tests run as part of its own pipeline.

The assignee now has the responsibility of keeping track how the pipeline of this new MR does.
If it succeeds, then it should be merged to master. If it fails, then the failing
tests should be checked and the reasons for failure should be fixed. Packages should
not be pinned within the .in files, just because tests are failing, unless there
is a very good reason for it. Packages pinned in the .in files should be regularly
revisited and if possible, unpinned.
