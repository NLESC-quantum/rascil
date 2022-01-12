## Run the Continuum Imaging Pipeline in a Kubernetes cluster

### Run in Minikube

1. Start minikube by mounting the directory that contains the MS. 
   The following command will mount it to `/mnt/data`

``` bash
minikube start --mount=true --mount-string="<local-data-dir>:/mnt/data"
```
where `local-data-dir` is the full path to the directory containing your data, which you want to mount.
You can add additional arguments to the command, e.g. `--driver=docker`, if needed.

2. Create the persistent volume and claim

``` bash
kubectl create -f pvc.yaml
```
The host path has to match the directory where your local one was mounted in minikube, i.e. /mnt/data
(Only `storageClassName: manual` works (`storageClassName: local` did not work for me (Gabi)))

If you are using k9s, type the following to list the persistent volume and
then the persistent volume claim, respectively: `:pv`; `:pvc`

3. Install the helm chart

``` bash 
helm install test ska-helm/dask -f values.yaml
```
Important: at the moment the `ska-helm/dask` chart does not contain the necessary
updates, which allow for the proper volume claims for workers. To use the updated code,
follow these steps:

- clone the SDP Helm Deployer Charts repository: 
`https://gitlab.com/ska-telescope/sdp/ska-sdp-helmdeploy-charts.git`
  
- check out the branch `orc-1156-update-dask-worker-with-volume-mount`:
`git checkout orc-1156-update-dask-worker-with-volume-mount`
  
- run the above helm command as follows:
`helm install test <path-to-ska-sdp-helmdeploy-charts>/charts/dask -f values.yaml`
  Do not forget to update the path!

Now you should have the Daks cluster running with one scheduler and two workers.
Update the values.yaml file if you require more or fewer workers. To see the pods
in k9s (if you changed to pv or pvc previously), type `:pod`.

Make sure you follow the instructions printed by the chart and run the port-forwarding commands.
These will also give you the link where you can access the Dask Dashboard.

4. Run the Continuum Imaging Pipeline

An example script is provided in `example_cip.sh`.
Port 8080 is where the scheduler is port-forwarded to. In the script, 
note the following arguments: 
`--ingest_msname /mnt/data/test.ms --logfile tmp.log` These specify where the 
code looks for the data. If your mounted directory is in a different place,
or the name of your `test.ms` MeasurementSet is different,
update the `--ingest_msname` path.
Note: `--logfile` cannot point to a specific directory. Because of the mounting, 
it only works if just the filename is given.

For the pipeline to work, you need to have a working RASCIL python environment.
Activate the environment, which contains the necessary python packages, and
run the following command:

``` bash
MALLOC_TRIM_THRESHOLD_=0 RASCIL_DASK_SCHEDULER=${DASK_SCHEDULER}:8080 sh example_cip.sh | tee -a example_cip.log
```
The code should print some logs on the terminal window, and you should
also be able to see how the pipeline is running in the Daks Dashboard.

5. Clean up

- Once you are finished, stop the port forwarding:
```bash
> fg
ctrl+c
```
(repeat until all port forwarding jobs are stopped)

- uninstall the chart: `helm uninstall test`

- remove the persistent volume and claim: `kubectl delete -f pvc.yaml`

- stop minikube