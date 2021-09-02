.. _rascil_release_process:

Build and Release process
*************************

Automatic builds
^^^^^^^^^^^^^^^^

RASCIL is built automatically via a GitLab CI pipeline, which can be triggered by:

    - on schedule
    - commit to any branch
    - merge/commit to master
    - a tag is pushed to the repository

The following stages/jobs run, depending on the trigger mechanism:

    - on schedule: the ``compile_requirements`` job runs, whose sole purpose is to regularly update the
        requirements files with the latest package versions. It also runs the ``.post`` stage.

    - commit to a branch: it runs the ``linting`` and ``test`` stages, as well as the ``prepost`` and ``.post`` ones.
        The latter two creates and posts the ``ci_metrics`` data.

    - merge/commit to master:
        * ``linting``, and ``test`` stages run
        * ``build`` stage runs with the ``data`` and ``build_package`` jobs. The first builds and saves the RASCIL data
          to GitLab, while the second builds the RASCIL python package for later consumption
        * the ``publish`` stage's ``docker_latest`` job runs, which builds, tags and publishes the latest docker images
          to the Central Artefact Repository. This stage also runs the ``pages`` job, which publishes the
          documentation and rebuilds the data.
        * ``prepost`` and ``.post`` stages run

    - commit tag: tagging the repository is manual (see below), which triggers the following parts of the pipeline
        * ``linting`` stage
        * ``build`` stage's ``build_package`` job, which builds the RASCIL python package
        * ``publish`` stage's ``publish_to_car`` and ``docker_release`` jobs. The first publishes the python package,
          while the second publishes the release-tagged (i.e. tagged with the package version) docker image
          to the Central Artefact Repository
        * ``.post`` stage

The above process makes sure that new code is automatically tested at
every point of the development process, and that the correct version
of the python package and the docker images are published with the
appropriate tag and at the appropriate time.

Releasing a new version
^^^^^^^^^^^^^^^^^^^^^^^

The release process:

* Overall based on: https://developer.skao.int/ and in particular https://developer.skao.int/en/latest/tools/software-package-release-procedure.html
* Use semantic versioning: https://semver.org
* Follow the packaging process in: https://packaging.python.org/tutorials/packaging-projects/

The release of a new package happens in two stages:

* a release tag is pushed to the repository (manually by a maintainer)
* the CI pipeline's relevant stages publish the new package.

Note: while commits are allowed directly to master by maintainers of the repository,
this should not be used as an option, but rather update the code via Merge Requests.
This is only allowed for releasing a new version of the package.


Steps:
------

 * Ensure that the current master builds on GitLab: https://gitlab.com/ska-telescope/external/rascil/-/pipelines
 * Decide whether a release is warranted and what semantic version number it should be: https://semver.org
 * Check if the documentation has been updated. If not, create a new branch, update the documentation,
   create a merge request and merge that to master (after approval).
 * Check out master and pull the latest version of it.
 * Update CHANGELOG.md for the relevant changes in this release, putting newer description at the top.
 * Commit the changes (do not push!)
 * Bump the version using the Makefile::

    make release-[patch||minor||major]

   Note: ``bumpver`` needs to be installed.
   This step automatically commits the new version tag to the repository.
 * Review the pipeline build for success
 * Create a new virtualenv and try the install by using pip3 install rascil::

        virtualenv test_env
        . test_env/bin/activate
        pip3 install --index-url=https://artefact.skao.int/repository/pypi-all/simple rascil
        python3
        >>> import rascil
