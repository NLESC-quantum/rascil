"""
Tests to be run from within the jupyter pod.
These check that the Dask cluster's elements
can connect to each other.

Used in the CI pipeline (ci_test.yaml)
"""
import os
from distributed import Client


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
