
Running RASCIL under docker
***************************

For some of the steps below it is helpful to have the RASCIL code tree available. Use::

   git clone https://gitlab.com/ska-telescope/rascil
   cd rascil

Running on existing docker images
---------------------------------

The RASCIL Dockerfiles are in a separate repository at https://gitlab.com/ska-telescope/rascil-docker.

The docker images for RASCIL are on nexus.engageska-portugal.pt at::

    nexus.engageska-portugal.pt/rascil-docker/rascil-base
    nexus.engageska-portugal.pt/rascil-docker/rascil-full
    nexus.engageska-portugal.pt/rascil-docker/rascil-notebook
    nexus.engageska-portugal.pt/rascil-docker/rascil-ci-checker

The first does not have the RASCIL test data but is smaller in size (2GB vs 4GB). However, for many of the tests
and demonstrations the test data is needed.

To run RASCIL with your home directory available inside the image::

    docker run -it --volume $HOME:$HOME nexus.engageska-portugal.pt/rascil-docker/rascil-full

Now let's run an example. First it simplifies using the container if we do not
try to write inside the container, and that's why we mapped in our $HOME directory.
So to run the /rascil/examples/scripts/imaging.py script, we first change directory
to the name of the HOME directory, which is the same inside and outside the
container, and then give the full address of the script inside the container. This time
we will show the prompts from inside the container::

     % docker run -p 8888:8888 -v $HOME:$HOME -it nexus.engageska-portugal.pt/rascil-docker/rascil-full
     rascil@d0c5fc9fc19d:/rascil$ cd /<your home directory>
     rascil@d0c5fc9fc19d:/<your home directory>$ python3 /rascil/examples/scripts/imaging.py
     ...
     rascil@d0c5fc9fc19d:/<your home directory>$ ls -l imaging*.fits
     -rw-r--r-- 1 rascil rascil 2102400 Feb 11 14:04 imaging_dirty.fits
     -rw-r--r-- 1 rascil rascil 2102400 Feb 11 14:04 imaging_psf.fits
     -rw-r--r-- 1 rascil rascil 2102400 Feb 11 14:04 imaging_restored.fits

In this example, we change directory to an external location (my home directory in this case,
use yours instead), and then we run the script using the absolute path name inside the container.

Running notebooks
-----------------

We also want to be able to run jupyter notebooks inside the container::

    docker run -it -p 8888:8888 --volume $HOME:$HOME nexus.engageska-portugal.pt/rascil-docker/rascil-full
    cd /<your home directory>
    jupyter notebook --no-browser --ip 0.0.0.0  /rascil/examples/notebooks/

The juptyer server will start and output possible URLs to use::

    [I 14:08:39.041 NotebookApp] Serving notebooks from local directory: /rascil/examples/notebooks
    [I 14:08:39.041 NotebookApp] The Jupyter Notebook is running at:
    [I 14:08:39.042 NotebookApp] http://d0c5fc9fc19d:8888/?token=f050f82ed0f8224e559c2bdd29d4ed0d65a116346bcb5653
    [I 14:08:39.042 NotebookApp]  or http://127.0.0.1:8888/?token=f050f82ed0f8224e559c2bdd29d4ed0d65a116346bcb5653
    [I 14:08:39.042 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
    [W 14:08:39.045 NotebookApp] No web browser found: could not locate runnable browser.

The 127.0.0.1 is the one we want. Enter this address in your local browser. You should see
the standard jupyter directory page.

Running the continuum_imaging_checker
-------------------------------------

A Docker image which runs the :ref:`rascil_apps_continuum_imaging_checker` is available at
``nexus.engageska-portugal.pt/rascil-docker/rascil-ci-checker:latest``. It can be run
in both Docker and Singularity.

DOCKER
++++++

Pull the image::

    docker pull nexus.engageska-portugal.pt/rascil-docker/rascil-ci-checker:latest

Run the image with default entrypoint will display the help interface of the continuum_imaging_checker::

    docker run nexus.engageska-portugal.pt/rascil-docker/rascil-ci-checker:latest

Run the image with input FITS files::

    docker run -v ${PWD}:/myData \
        -e CLI_ARGS='--ingest_fitsname_restored /myData/my_restored.fits \
        --ingest_fitsname_residual /myData/my_residual.fits' \
        --rm nexus.engageska-portugal.pt/rascil-docker/rascil-ci-checker:latest

The above command will mount your current directory int `myData` on the container filesystem.
The code within the container will access your data files in this directory, so make sure, you
run it from the directory where your images you want to check are. The output files will
appear in the same directory on your local system. Update the ``CLI_ARGS`` string with the command
line arguments of the :ref:`rascil_apps_continuum_imaging_checker` code as needed.
Once the run finishes, the container will be automatically removed from the system
because of ``--rm`` in the above command.

SINGULARITY
+++++++++++

Pull the image and name it ``rascil-ci-checker.img``::

    singularity pull rascil-ci-checker.img docker://nexus.engageska-portugal.pt/rascil-docker/rascil-ci-checker:latest

Run the image with default entrypoint will display the help interface of the continuum_imaging_checker::

    singularity run rascil-ci-checker.img

Run the image with input FITS files::

    singularity run \
        --env CLI_ARGS='--ingest_fitsname_restored test-imaging-pipeline-dask_continuum_imaging_restored.fits \
            --ingest_fitsname_residual test-imaging-pipeline-dask_continuum_imaging_residual.fits' \
        rascil-ci-checker.img

Run it from the directory where your images you want to check are. The output files will
appear in the same directory. If the singularity image you downloaded is in a different path,
point to that path in the above command. Update the ``CLI_ARGS`` string with the command line
arguments of the :ref:`rascil_apps_continuum_imaging_checker` code as needed.

Providing input arguments from a file
+++++++++++++++++++++++++++++++++++++

You may create a file that contains the input arguments for the app. Here is an example of it,
called ``args.txt``::

    --ingest_fitsname_restored=/myData/test-imaging-pipeline-dask_continuum_imaging_restored.fits
    --ingest_fitsname_residual=/myData/test-imaging-pipeline-dask_continuum_imaging_residual.fits
    --check_source=True
    --plot_source=True

Make sure each line contains one argument, there is an equal sign between arg and its value,
and that there aren't any trailing white spaces in the lines. The paths to images and other input
files has to be the absolute path within the container. Here, we use the ``DOCKER`` example of
mounting our data into the ``/myData`` directory.

Then, calling ``docker run`` simplifies as::

    docker run -v ${PWD}:/myData \
    -e CLI_ARGS='@/myData/args.txt \
    --rm nexus.engageska-portugal.pt/rascil-docker/rascil-ci-checker:latest

Here, we assume that your custom args.txt file is also mounted together with the data into ``/myData``.
Provide the absolute path to that file when your run the above command.

You can use an args file to run the singularity version with same principles.

Running RASCIL as a cluster
---------------------------

The file docker-compose in the rascil-docker code tree provides a simple way to
create a local cluster of a Dask scheduler and a number of workers. First install
the rascil-docker code tree::

       git clone https://gitlab.com/ska-telescope/rascil-docker
       cd rascil-docker

The cluster is created using the docker-compose up command. To scale to e.g. 4 dask workers::

    docker-compose up -f docker-compose-base.yml --scale worker=4

The scheduler, 4 workers and a notebook should now be running. To connect to the cluster, run the
following into another window::

    docker run -it --network host --volume $HOME:$HOME nexus.engageska-portugal.pt/rascil-docker/rascil-full

Then at the docker prompt, do e.g.::

    cd /<your home directory>
    python3 /rascil/cluster_tests/ritoy/cluster_test_ritoy.py localhost:8786

A jupyter lab notebook is also started by this docker-compose. The URL will be output during the
initial set up, e.g.::

    notebook_1   | [I 15:17:05.681 NotebookApp] The Jupyter Notebook is running at:
    notebook_1   | [I 15:17:05.682 NotebookApp] http://notebook:8888/?token=0e77cf0e214fb0f5827b35fa5de8bbc5ebed6d4159e3d31e
    notebook_1   | [I 15:17:05.682 NotebookApp]  or http://127.0.0.1:8888/?token=0e77cf0e214fb0f5827b35fa5de8bbc5ebed6d4159e3d31e
    notebook_1   | [I 15:17:05.682 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).

Click on the 127.0.0.1 URL. We have used the jupyter lab interface instead of jupyter notebook interface
because the former allows control of Dask from the interface. This can be changed in the docker-compose.yml
file. Note also that the classic notebook interface can be selected at the lab interface.

If the RASCIL data is already locally available then the images can be built without data using a slightly
different compose file. This assumes that the environment variable RASCIL_DATA points to the
data::

    docker-compose --file docker-compose-base.yml up --scale worker=4

The scheduler, 4 workers and notebook should now be running and can be accessed as above.

CASA Measures Tables
--------------------

We use the CASA measures system for TAI/UTC corrections. These rely upon tables downloaded from NRAO.
It may happen that the tables become out ofdate. If so do the following at the command prompt inside a
docker image::

    rsync -avz rsync://casa-rsync.nrao.edu/casa-data/geodetic /var/lib/casacore/data


Singularity
-----------

`Singularity <https://sylabs.io/docs/>`_ can be used to load and run the docker images::

    singularity pull RASCIL-full.img docker://nexus.engageska-portugal.pt/rascil-docker/rascil-full
    singularity exec RASCIL-full.img python3 /rascil/examples/scripts/imaging.py

As in docker, don't run from the /rascil/ directory.

Inside a SLURM file singularity can be used by prefacing dask and python commands with "singularity exec". For example::

    ssh $host singularity exec /home/<your-name>/workspace/RASCIL-full.img dask-scheduler --port=8786 &
    ssh $host singularity exec /home/<your-name>/workspace/RASCIL-full.img dask-worker --host ${host} --nprocs 4 --nthreads 1  \
    --memory-limit 100GB $scheduler:8786 &
    CMD="singularity exec /home/<your-name>/workspace/RASCIL-full.img python3 ./cluster_test_ritoy.py ${scheduler}:8786 | tee ritoy.log"
    eval $CMD

Customisability
---------------

The docker images described here are ones we have found useful. However,
if you have the RASCIL code tree installed then you can also make your own versions
working from these Dockerfiles.

