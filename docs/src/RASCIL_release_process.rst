
Release process
***************

This is a reminder to the maintainers of how the release process is to be done.

* Overall based on: https://developer.skatelescope.org/ and in particular https://developer.skatelescope.org/en/latest/development/software_package_release_procedure.html
* Use semantic versioning: https://semver.org
* Follow the packaging process in: https://packaging.python.org/tutorials/packaging-projects/


Steps:
------

 * Ensure that the current master builds on GitLab: https://gitlab.com/timcornwell/rascil/pipelines
 * Decide whether a release is warranted and what semantic version number it should be: https://semver.org
 * Update CHANGELOG.md for the relevant changes in this release, putting newer description at the top.
 * Check if  the documentation been updated
 * Update setup.py for the new version number e.g. 0.1.6
 * Update README.md as appropriate
 * Goto rascil-docker, update the version number (e.g. to 0.1.6)
 * Tag the release e.g.::

        git tag -a v.0.1.6 -m "Docker files moved to separate repo"


 * Goto rascil, push the rascil changes to the master. This will trigger a build of rascil and then a build in
rascil-docker. The presence of the tag will trigger construction and publication of the pip files. At the moment the
pip files are also labelled by build info. The following should work::

        pip3 install --extra-index-url=https://nexus.engageska-portugal.pt/repository/pypi/simple/ "rascil>=0.1.7"


 * Review the pipeline build for success
 * Create a new virtualenv and try the install by using pip3 install rascil::

        virtualenv test_env
        . test_env/bin/activate
        pip3 install --extra-index-url=https://nexus.engageska-portugal.pt/repository/pypi/simple/ "rascil>=0.1.7"
        python3
        >>> import rascil


 * Reset the version numbers both in rascil and rascil-docker to the next e.g. 0.1.17b0
 * Make a stable release of the docker files from rascil-docker.
