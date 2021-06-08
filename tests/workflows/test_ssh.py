from dask.distributed import Client, SSHCluster
import dask_memusage

cluster = SSHCluster(
    [
        "openhpc-compute-0",
        "openhpc-compute-1",
        "openhpc-compute-2",
        "openhpc-compute-3",
        "openhpc-compute-4",
        "openhpc-compute-5",
        "openhpc-compute-6",
        "openhpc-compute-7",
    ],
    connect_options={"known_hosts": None},
    worker_options={"nthreads": 1},
    scheduler_options={"port": 0, "dashboard_address": ":8797"},
)
client = Client(cluster)
dask_memusage.install(cluster.scheduler, "mem.csv")
