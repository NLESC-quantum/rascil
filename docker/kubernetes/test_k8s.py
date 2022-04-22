"""
Test the Kubernetes cluster set up,
i.e. whether the right pods are installed
and running, and if they connect to the
correct PersistentVolumeClaim.

Used in the CI pipeline (ci_test.yaml)
"""
import os
from kubernetes import client, config

config.load_kube_config()

NAMESPACE = os.environ.get("NAMESPACE", None)

# This information is from docker/kubernetes/values.yaml
# and from the pvc definition in docker/kubernetes/Makefile
WORKER_REPLICAS = 2
RASCIL_PVC = "pvc-rascil-cip"
MOUNT_PATH = "/mnt/data"


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
