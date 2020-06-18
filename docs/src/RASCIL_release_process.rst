
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

 * Goto rascil, push the rascil changes to the master. This will trigger a build of rascil and then a
build in rascil-docker. The presence of the tag will trigger construction aand publication of the pip files
 * Review the pipeline build for success
 * Create a new virtualenv and try the install by using pip3 install rascil::

        virtualenv test_env
        . test_env/bin/activate
        pip install rascil
        python3
        >>> import rascil

 * Reset the version numbers both in rascil and rascil-docker to the next e.g. 0.1.17b0
