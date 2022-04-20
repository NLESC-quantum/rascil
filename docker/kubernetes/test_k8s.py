"""
Kubernetes cluster set up (i.e. installed pods) tests
are run from outside a pod, directly from the CI pipeline.
These are marked "outside".

Tests that are marked as "inside" are run
from within the jupyter pod. These check that the
Dask cluster's elements can connect to each other.

Used in the CI pipeline (ci_test.yaml)
"""
import os
import pytest

from kubernetes import client, config
from distributed import Client

config.load_config()

NAMESPACE = os.environ.get("NAMESPACE", None)

# This information is from docker/kubernetes/values.yaml
# and from the pvc definition in docker/kubernetes/Makefile
WORKER_REPLICAS = 2
RASCIL_PVC = "pvc-rascil-cip"
MOUNT_PATH = "/mnt/data"


@pytest.mark.outside
def test_check_pods():
    """
    Test that the right number of pods have been started.
    The number of worker pods matches the number of
    replicas requested in values.yaml
    Also check that the pods are in "Running" state.
    """
    kube_client = client.CoreV1Api()

    pods = kube_client.list_namespaced_pod(NAMESPACE)
    jupyter = [pod for pod in pods.items if "jupyter" in pod.metadata.name]
    scheduler = [pod for pod in pods.items if "scheduler" in pod.metadata.name]
    workers = [pod for pod in pods.items if "worker" in pod.metadata.name]

    # 1 scheduler, 1 jupyter, x workers
    assert len(jupyter) == 1
    assert len(scheduler) == 1
    assert len(workers) == WORKER_REPLICAS

    for pod in jupyter + scheduler + workers:
        assert pod.__getattribute__("status").phase == "Running"


@pytest.mark.outside
def test_volume_mount():
    """
    The jupyter pod and all the Dask worker pods
    have the correct volume mounted through the
    pvc-rascil-cip PersistentVolumeClaim
    """
    kube_client = client.CoreV1Api()
    pods = kube_client.list_namespaced_pod(NAMESPACE)

    jupyter = [pod for pod in pods.items if "jupyter" in pod.metadata.name]
    workers = [pod for pod in pods.items if "worker" in pod.metadata.name]

    for pod in jupyter + workers:
        # containers[0] --> each pod has a single container in it
        volumes = pod.__getattribute__("spec").containers[0].volume_mounts
        rascil_volume = [volume for volume in volumes if volume.name == RASCIL_PVC]

        assert len(rascil_volume) == 1
        # volume mount location inside container is /mnt/data
        assert rascil_volume[0].mount_path == MOUNT_PATH


@pytest.mark.outside
def test_pvc():
    """
    PersistentVolumeClaim for the RASCIL deployment has
    been created, and it is bound to a PersistentVolume
    """
    kube_client = client.CoreV1Api()

    pvc = kube_client.list_namespaced_persistent_volume_claim(NAMESPACE)
    rascil_pvc = [p for p in pvc.items if p.metadata.name == RASCIL_PVC]

    assert len(rascil_pvc) == 1
    assert rascil_pvc[0].status.phase == "Bound"


@pytest.mark.inside
def test_dask_connection():
    """
    Run from the jupyter pod.

    Connect to the Dask scheduler and test that the scheduler
    has the correct workers connected to it (based on IP address).

    We pass in the worker IPs via an environment variable
    (see test_k8s_jupyter make target in docker/kubernetes/Makefile).

    This test also confirms that the jupyter pod can see and connect
    to the Dask scheduler.
    """
    worker_ips = os.environ["WORKER_IPS"].split(" ").sort()
    dask_client = Client("rascil-dask-scheduler:8786")
    workers_on_scheduler = dask_client.scheduler_info()["workers"]
    result_ips = [v["host"] for k, v in workers_on_scheduler.items()].sort()
    assert result_ips == worker_ips
