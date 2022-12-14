.. _rascil_GALAHAD_install:

Installation of RASCIL on galahad
=================================

RASCIL is well-suited to running on galahad. Installation should be straightforward.
We strongly recommend the use of a python virtual environment. Be sure to load the
bare python base module and the gcc modules (in the supported versions) before 
installing RASCIL, e.g. ::

    module load python39base gcc920

Follow the generic installation steps.

We recommend that RASCIL be installed on one of the preferred storage
systems e.g. /share/nas/<your-login>/rascil/

If you are using singularity containers, you will probably need to put the
singularity cache somewhere other than your home directory::

    mkdir /share/nas/<yourname>/.singularity
    export SINGULARITY_CACHEDIR=/share/nas/<yourname>/.singularity
    singularity pull RASCIL-full.img docker://timcornwell/rascil-full

.. _feedback: mailto:realtimcornwell@gmail.com
