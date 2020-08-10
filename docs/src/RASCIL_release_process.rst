
Release process
***************

The intention of the RASCIL release process is that:

 * A commit of any branch, including the master, results in a full build.
 * The full build includes all the unittests and the documentation.
 * The documentation is published only for builds of the master.
 * The master is only updated via merge_requests
 * New builds of the master occur with every commit or merge request.
 * Releases are made of successful builds as appropriate, typically every few weeks.
 * The pip file is updated on every successful build of the master. The tag will be e.g. 0.1.9_b38a820d
 * If this is a new release the pip tag will be just the tag e.g. 0.1.9
 * The latest versions of the docker files will be updated on every successful build of the master.
 * If this is a new release all the docker images will also be tagged stable.

The release process as follows:

* Overall based on: https://developer.skatelescope.org/ and in particular https://developer.skatelescope.org/en/latest/development/software_package_release_procedure.html
* Use semantic versioning: https://semver.org
* Follow the packaging process in: https://packaging.python.org/tutorials/packaging-projects/

Steps:
------

 * Ensure that the current master builds on GitLab: https://gitlab.com/ska-telescope/rascil/-/pipelines
 * Decide whether a release is warranted and what semantic version number it should be: https://semver.org
 * Update CHANGELOG.md for the relevant changes in this release, putting newer description at the top.
 * Check if  the documentation been updated
 * Update setup.py for the new version number e.g. 0.1.6
 * Update README.md as appropriate
 * Goto rascil-docker, update the version number (e.g. to 0.1.6)
 * Tag the release e.g.::

        git tag -a 0.1.6 -m "Docker files moved to separate repo"


 * Goto rascil, push the rascil changes to the master. This will trigger a build of rascil and then a build in
    rascil-docker. The pip file can be installed as follows::

        pip3 install --extra-index-url=https://nexus.engageska-portugal.pt/repository/pypi/simple/ rascil


 * Review the pipeline build for success
 * Create a new virtualenv and try the install by using pip3 install rascil::

        virtualenv test_env
        . test_env/bin/activate
        pip3 install --extra-index-url=https://nexus.engageska-portugal.pt/repository/pypi/simple/ rascil
        python3
        >>> import rascil

 * Reset the version numbers both in rascil and rascil-docker to the next e.g. 0.1.9b0
