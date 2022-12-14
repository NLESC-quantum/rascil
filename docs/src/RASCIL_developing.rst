.. _rascil_developing:

Developing in RASCIL
********************

Use the SKA Python Coding Guidelines (http://developer.skatelescope.org/en/latest/development/python-codeguide.html).

We recommend using a tool to help ensure PEP 8 compliance. PyCharm does a good job at this and other code quality
checks.

Process
=======

- Use git to make a local clone of the Github respository::

   git clone https://gitlab.com/ska-telescope/rascil

- Make a branch. Use a descriptive name e.g. feature_improved_gridding, bugfix_issue_666
- Make whatever changes are needed, including documentation.
- Always add appropriate test code in the tests directory.
- Consider adding to the examples area.
- Push the branch to gitlab. It will then be automatically built and tested on gitlab: https://gitlab.com/ska-telescope/rascil/-/pipelines
- Once it builds correctly, submit a merge request.


Design
======

The RASCIL has been designed in line with the following principles:

+ Data are held in Classes.
+ The Data Classes correspond to familiar concepts in radio astronomy packages e.g. visibility, gaintable, image.
+ The data members of the Data Classes are directly accessible by name e.g. .data, .name, .phasecentre.
+ Direct access to the data members is envisaged.
+ There are no methods attached to the data classes apart from variant constructors as needed.
+ Standalone, stateless functions are used for all processing.

Additions and changes should adhere to these principles.

Submitting code
===============

RASCIL is part of the SKA telescope organisation on GitLab. https://gitlab.com/ska-telescope/rascil.git. 

We welcome merge requests submitted via GitLab. Please note that we use Black to keep the python
code style in good shape. The first step in the CI pipeline checks that the code complies with
black formatting style, and will fail if that is not the case.