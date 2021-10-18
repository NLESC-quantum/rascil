# Dockerfiles for RASCIL

These Dockerfiles construct RASCIL docker images working from the RASCIL master. 
They can be found in the `docker` directory and its subdirectories.
These currently support building, pushing, and tagging images. The images are named
as specified in the `release` file of the docker image directory,
and tagged by the RASCIL version stored in `rascil/version.py`. 

There are various directories for docker files:

- rascil-base: A minimal RASCIL, without data
- rascil-full: Base with data
- rascil-notebook: Supports running jupyter notebook
- rascil-imaging-qa: Runs the Continuum Imaging QA tool

## Automatic publishing

The docker images are automatically built by the CI pipeline.
The `latest` versions are pushed to the [Central Artifact Repository](https://artefact.skao.int/#browse/browse:docker-all)
(CAR) upon merge to the master branch, while the release version 
(tagged with the RASCIL version number) are pushed when a commit tag
is pushed to the repository

Note: at the moment, `latest` versions are NOT accessible from the CAR.
To find out what versions you can download, look for the relevant 
RASCIL docker image in the [Central Artifact Repository](https://artefact.skao.int/#browse/browse:docker-all)

## Build, push, and tag a set of Dockerfiles

If you want to build an image yourself, follow these steps:

- `cd` into one of the subdirectories
- Build the image with `make build`

Other useful make commands :

- `push` pushes the images to the docker registry 
- `push_latest` pushes the `:latest` tag
- `push_version` pushes a version tag without the git SHA 
  
Useful make commands that can be run from the `docker` directory:

- `build_all_latest` builds, and tags as latest, all the images
- `rm_all` removes all the images
- `ls_all` lists all the images

## Test the images

The `docker/Makefile` contains commands for testing all the images. 
These write results into the host /tmp area. For docker:

- make test_base
- make test_full
- make test_notebook
- make test_imaging_qa

And for singularity:

- make test_base_singularity
- make test_full_singularity
- make test_notebook_singularity
- make test_imaging_qa_singularity

## Generic RASCIL images

### rascil-base and rascil-full

The base and full images are available at::

    artefact.skao.int/rascil-base
    artefact.skao.int/rascil-full

`rascil-base` does not have the RASCIL test data but is smaller in size. 
However, for many of the tests and demonstrations the test data is needed, which are included in `rascil-full`.

To run RASCIL with your home directory available inside the image::

    docker run -it --volume $HOME:$HOME artefact.skao.int/rascil-full

Now let's run an example. First it simplifies using the container if we do not
try to write inside the container, and that's why we mapped in our $HOME directory.
So to run the /rascil/examples/scripts/imaging.py script, we first change directory
to the name of the HOME directory, which is the same inside and outside the
container, and then give the full address of the script inside the container. This time
we will show the prompts from inside the container::

     % docker run -p 8888:8888 -v $HOME:$HOME -it artefact.skao.int/rascil-full
     rascil@d0c5fc9fc19d:/rascil$ cd /<your home directory>
     rascil@d0c5fc9fc19d:/<your home directory>$ python3 /rascil/examples/scripts/imaging.py
     ...
     rascil@d0c5fc9fc19d:/<your home directory>$ ls -l imaging*.fits
     -rw-r--r-- 1 rascil rascil 2102400 Feb 11 14:04 imaging_dirty.fits
     -rw-r--r-- 1 rascil rascil 2102400 Feb 11 14:04 imaging_psf.fits
     -rw-r--r-- 1 rascil rascil 2102400 Feb 11 14:04 imaging_restored.fits

In this example, we change directory to an external location (my home directory in this case,
use yours instead), and then we run the script using the absolute path name inside the container.

## RASCIL Notebooks

The docker image to use with RASCIL Jupyter Notebooks is::

    artefact.skao.int/rascil-notebook

Run Jupyter Notebooks inside the container::

    docker run -it -p 8888:8888 --volume $HOME:$HOME artefact.skao.int/rascil-full
    cd /<your home directory>
    jupyter notebook --no-browser --ip 0.0.0.0  /rascil/examples/notebooks/

The Juptyer server will start and output possible URLs to use::

    [I 14:08:39.041 NotebookApp] Serving notebooks from local directory: /rascil/examples/notebooks
    [I 14:08:39.041 NotebookApp] The Jupyter Notebook is running at:
    [I 14:08:39.042 NotebookApp] http://d0c5fc9fc19d:8888/?token=f050f82ed0f8224e559c2bdd29d4ed0d65a116346bcb5653
    [I 14:08:39.042 NotebookApp]  or http://127.0.0.1:8888/?token=f050f82ed0f8224e559c2bdd29d4ed0d65a116346bcb5653
    [I 14:08:39.042 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
    [W 14:08:39.045 NotebookApp] No web browser found: could not locate runnable browser.

The 127.0.0.1 is the one we want. Enter this address in your local browser. You should see
the standard Jupyter directory page.

## Images of RASCIL applications

### Continuum imaging QA tool (a.k.a imaging_qa)

[imaging_qa](Documentation: https://ska-telescope.gitlab.io/external/rascil/apps/imaging_qa.html) finds compact sources in a continuum image and compares them 
to the sources used in the simulation, thus revealing the quality of the imaging.

####DOCKER

Pull the image:

`docker pull artefact.skao.int/rascil-imaging-qa:latest`

Run the image:

```
docker run -v ${PWD}:/myData -e DOCKER_PATH=${PWD} \
    -e CLI_ARGS='--ingest_fitsname_restored /myData/my_restored.fits \
    --ingest_fitsname_residual /myData/my_residual.fits' \
    --rm artefact.skao.int/rascil-imaging-qa:latest
```
Run it from the directory where your images you want to check are. The output files will
appear in the same directory. Update the `CLI_ARGS` string with the command line arguments
of the imaging QA code as needed. `DOCKER_PATH` is used to extract the path
of the output files the app produced in your local machine, not in the docker container. This
is used for generating the output file index files.

####SINGULARITY

Pull the image:

`singularity pull rascil-imaging-qa.img docker://artefact.skao.int/rascil-imaging-qa:latest`

Run the image:

```
singularity run \
    --env CLI_ARGS='--ingest_fitsname_restored test-imaging-pipeline-dask_continuum_imaging_restored.fits \
        --ingest_fitsname_residual test-imaging-pipeline-dask_continuum_imaging_residual.fits' \
    rascil-imaging-qa.img
```
Run it from the directory where your images you want to check are. The output files will
appear in the same directory. If the singularity image you downloaded is in a different path, 
point to that path in the above command. Update the `CLI_ARGS` string with the command line arguments
of the imaging QA code as needed.

####Providing input arguments from a file

You may create a file that contains the input arguments for the app. Here is an example of it,
called `args.txt`::
    
    --ingest_fitsname_restored=/myData/test-imaging-pipeline-dask_continuum_imaging_restored.fits
    --ingest_fitsname_residual=/myData/test-imaging-pipeline-dask_continuum_imaging_residual.fits
    --check_source=True
    --plot_source=True

Make sure each line contains one argument, there is an equal sign between arg and its value,
and that there aren't any trailing white spaces in the lines (and no empty lines). 
The paths to images and other input files has to be the absolute path within the container. 
Here, we use the `DOCKER` example of mounting our data into the `/myData` directory.

Then, calling `docker run` simplifies as::

    docker run -v ${PWD}:/myData  -e DOCKER_PATH=${PWD} -e CLI_ARGS='@/myData/args.txt' \
    --rm artefact.skao.int/rascil-imaging-qa:latest

Here, we assume that your custom args.txt file is also mounted together with the data into ``/myData``. 
Provide the absolute path to that file when your run the above command.

You can use an args file to run the singularity version with same principles, baring in mind
that singularity will automatically mount your filesystem into the container with paths
matching those on your system.

##Running RASCIL as a cluster

The following methods of running RASCIL as a cluster, will provide a set of 
docker-based environments, which host a Dask scheduler, various Dask workers 
(numbers can be customized), and a Jupyter lab notebook, which directly
connects to the scheduler.

###Kubernetes

RASCIL can be run as a cluster in [Kubernetes](https://kubernetes.io/) using 
[helm](https://helm.sh/) and [kubectl](https://kubernetes.io/docs/reference/kubectl/overview/) 
(you need to have these two installed). If you want to run it in a local developer environment 
(e.g. a laptop), we recommend using [Minikube](https://minikube.sigs.k8s.io/docs/start/).

A custom `values.yaml` files is provided in 
[/rascil/docker/kubernetes](https://gitlab.com/ska-telescope/external/rascil/-/tree/master/docker/kubernetes/values.yaml).
It is meant to be used with a custom Dask Helm chart maintained by SKA developers, 
hosted in a [GitLab repository](https://gitlab.com/ska-telescope/sdp/ska-sdp-helmdeploy-charts/-/tree/master/chart-repo).
The documentation and details of the SKA Dask Helm chart can be found at
https://developer.skao.int/projects/ska-sdp-helmdeploy-charts/en/latest/charts/dask.html.

You can modify the `values.yaml` file, if needed, e.g. you can change the number of 
worker replicas, or the docker image used (e.g. the version that should be run).

Start Minikube and add the helm repository::

    helm repo add ska-helm https://gitlab.com/ska-telescope/sdp/ska-sdp-helmdeploy-charts/-/raw/master/chart-repo
    helm repo update

`cd` into the `/rascil/docker/kubernetes` directory and install the RASCIL cluster::

    helm install test ska-helm/dask -f values.yaml

Instructions on how to connect to the Dask dashboard and the Jupyter lab notebook are printed in the screen,
please follow those. You can follow the deployment process and access logs using `kubectl` or via
[`k9s`](https://k9scli.io/).

To uninstall the chart and clean out all pods, run::

    helm uninstall test

Note: this will remove changes you might have made in the Jupyter notebooks.


## CASA Measures Tables

We use the CASA measures system for TAI/UTC corrections. These rely upon tables downloaded from NRAO.
It may happen that the tables become out of date. If so do the following at the command prompt inside a
docker image::

    rsync -avz rsync://casa-rsync.nrao.edu/casa-data/geodetic /var/lib/casacore/data


##Singularity

`Singularity <https://sylabs.io/docs/>`_ can be used to load and run the docker images::

    singularity pull RASCIL-full.img docker://artefact.skao.int/rascil-full
    singularity exec RASCIL-full.img python3 /rascil/examples/scripts/imaging.py

As in docker, don't run from the /rascil/ directory.

Inside a SLURM file singularity can be used by prefacing dask and python commands with "singularity exec". For example::

    ssh $host singularity exec /home/<your-name>/workspace/RASCIL-full.img dask-scheduler --port=8786 &
    ssh $host singularity exec /home/<your-name>/workspace/RASCIL-full.img dask-worker --host ${host} --nprocs 4 --nthreads 1  \
    --memory-limit 100GB $scheduler:8786 &
    CMD="singularity exec /home/<your-name>/workspace/RASCIL-full.img python3 ./cluster_test_ritoy.py ${scheduler}:8786 | tee ritoy.log"
    eval $CMD

##Customisability

The docker images described here are ones we have found useful. However,
if you have the RASCIL code tree installed then you can also make your own versions
working from these Dockerfiles.

## Important updates

Starting with version 0.3.0, RASCIL is installed as a package into the docker images and
the repository is not cloned anymore. Hence, every python script 
(except the ones in the `examples` directory) within the image has to be
called with the ``-m`` switch in the following format, when running within the docker container, e.g.:

.. code-block:: 

    python -m rascil.apps.rascil_advise <args>