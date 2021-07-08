"""Functions for monitoring performance

These functions can be used to write various configuration and performance information to
JSON files for subsequent analysis. These are intended to be used by apps such as rascil-imager::

    parser = cli_parser()
    args = parser.parse_args()
    performance_environment(args.performance_file, mode="w")
    performance_store_dict(args.performance_file, "cli_args", vars(args), mode="a")
    performance_store_dict(args.performance_file, "dask_profile", dask_info, mode="a")
    performance_dask_configuration(args.performance_file, mode='a')

"""

__all__ = [
    "performance_store_dict",
    "performance_qa_image",
    "performance_dask_configuration",
    "performance_read",
    "git_hash",
]

import csv
import json
import logging
import os
import socket
import subprocess
from datetime import datetime

import numpy

import dask

from rascil.processing_components.image.operations import qa_image

log = logging.getLogger("rascil-logger")


def git_hash():
    """Get the hash for this git repository.

    Requires that the code tree was created using git

    :return: string or "unknown"
    """
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"])
    except Exception as excp:
        log.info(excp)
        return "unknown"


def performance_read(performance_file):
    """Read the performance file

    :param performance_file:
    :return: Dictionary
    """
    if performance_file is None:
        raise ValueError("performance_file is set to None: cannot read")

    try:
        with open(performance_file, "r") as file:
            performance = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"performance file {performance_file} does not exist")

    # Now read the memory file and merge into the performance data
    mem_file = performance_file.replace(".json", ".csv")
    try:
        mem = performance_read_memory_data(mem_file)
        performance = performance_merge_memory(performance, mem)
    except FileNotFoundError:
        log.warning(f"No memory file {mem_file} found ")
        pass
    return performance


def performance_blockvisibility(bvis):
    """Get info about the blockvisibility

    This works on a single blockvisibility because we probably want to send this function to
    the cluster instead of bringing the data back

    :param bvis:
    :return: bvis info as a dictionary
    """
    bv_info = {
        "number_times": bvis.blockvisibility_acc.ntimes,
        "number_baselines": len(bvis.baselines),
        "nchan": bvis.blockvisibility_acc.nchan,
        "npol": bvis.blockvisibility_acc.npol,
        "polarisation_frame": bvis.blockvisibility_acc.polarisation_frame.type,
        "nvis": bvis.blockvisibility_acc.ntimes
        * len(bvis.baselines)
        * bvis.blockvisibility_acc.nchan
        * bvis.blockvisibility_acc.npol,
        "size": bvis.nbytes,
    }
    return bv_info


def performance_environment(performance_file, indent=2, mode="a"):
    """Write the current processing environment to JSON file

    :param performance_file: The (JSON) file to which the environment is to be written
    :param indent: Number of characters indent in performance file
    :param mode: Writing mode: 'w' or 'a' for write and append
    """
    if performance_file is not None:
        info = {
            "git": str(git_hash()),
            "cwd": os.getcwd(),
            "hostname": socket.gethostname(),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        performance_store_dict(
            performance_file, "environment", info, indent=indent, mode=mode
        )


def performance_dask_configuration(performance_file, rsexec, indent=2, mode="a"):
    """Get selected Dask configuration info and write to performance file

    :param performance_file: The (JSON) file to which the environment is to be written
    :param rsexec: rsexecute passed in to avoid dependency
    :param indent: Number of characters indent in performance file
    :param mode: Writing mode: 'w' or 'a' for write and append
    """

    if performance_file is not None:
        if rsexec.using_dask:
            if (
                rsexec.client is not None
                and rsexec.client.cluster is not None
                and rsexec.client.cluster.scheduler_info is not None
            ):
                # LocalCluster or similar
                workers = rsexec.client.cluster.workers
                info = {
                    "client": str(rsexec.client),
                    "nworkers": len(workers),
                    "nthreads": int(
                        numpy.sum([workers[worker].nthreads for worker in workers])
                    ),
                    "tcp_timeout": dask.config.get("distributed.comm.timeouts.tcp"),
                    "connect_timeout": dask.config.get(
                        "distributed.comm.timeouts.connect"
                    ),
                }

            else:
                # Distributed
                workers = rsexec.client.scheduler_info()["workers"]

                info = {
                    "client": str(rsexec.client),
                    "nworkers": len(workers),
                    "nthreads": int(
                        numpy.sum([workers[worker]["nthreads"] for worker in workers])
                    ),
                }
        else:
            info = {
                "client": "",
                "nworkers": 0,
                "nthreads": 0,
            }

        performance_store_dict(
            performance_file, "dask_configuration", info, indent=indent, mode=mode
        )


def performance_qa_image(performance_file, key, im, indent=2, mode="a"):
    """Store image qa in a performance file

    :param performance_file: The (JSON) file to which the environment is to be written
    :param key: Key to use for the configuration info e.g. "restored"
    :param im: Image for which qa is to be calculated and written
    :param indent: Number of characters indent in performance file
    :param mode: Writing mode: 'w' or 'a' for write and append
    """

    if performance_file is not None:
        qa = qa_image(im)
        performance_store_dict(performance_file, key, qa.data, indent=indent, mode=mode)


def performance_store_dict(performance_file, key, s, indent=2, mode="a"):
    """Store dictionary in a file using json

    :param performance_file: The (JSON) file to which the environment is to be written
    :param key: Key to use for the configuration info e.g. "restored"
    :param s: dictionary to be written
    :param indent: Number of characters indent in performance file
    :param mode: Writing mode: 'w' or 'a' for write and append
    """
    if performance_file is not None:
        if mode == "w":
            with open(performance_file, mode) as file:
                s = json.dumps({key: s}, indent=indent)
                file.write(s)
        elif mode == "a":
            try:
                with open(performance_file, "r") as file:
                    previous = json.load(file)
                    previous[key] = s
                with open(performance_file, "w") as file:
                    s = json.dumps(previous, indent=indent)
                    file.write(s)
            except FileNotFoundError:
                with open(performance_file, "w") as file:
                    s = json.dumps({key: s}, indent=indent)
                    file.write(s)


def performance_read_memory_data(memory_file):
    """Get the memusage data.

    An example of the csv file:
    task_key,min_memory_mb,max_memory_mb
    create_blockvisibility_from_ms-6d4df60d-244b-4a45-8dca-a7d96b676455,219.80859375,7651.37109375
    getitem-ab6cb10a048f6d5efce69194feafa125,0,0
    performance_blockvisibility-2dfe2b3a-e160-4724-a5e6-aed82bf0721c,0,0
    create_blockvisibility_from_ms-724c98e9-279b-44ef-92d6-06e689b037a2,223.72265625,7642.13671875

    The task_key is split into task and key. The memory values are converted to GB.

    :param memory_file: Dictionary containing sequences of maximum and minimum memory for each function sampled
    :return:
    """
    functions = list()
    keys = list()
    max_mem = list()
    min_mem = list()

    with open(memory_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # task_key,min_memory_mb,max_memory_mb
            functions.append(row["task_key"].split("-")[0])
            keys.append(row["task_key"].split("-")[1])
            max_mem.append(2 ** -10 * float(row["max_memory_mb"]))
            min_mem.append(2 ** -10 * float(row["min_memory_mb"]))
    mem = {
        "functions": numpy.array(functions),
        "keys": numpy.array(keys),
        "max_memory": numpy.array(max_mem),
        "min_memory": numpy.array(min_mem),
    }

    return mem


def performance_merge_memory(performance, mem):
    """Merge memory data per function into performance data

    The memory usage information comes from the optional use of the dask-memusage
    scheduler plugin

    :param performance: Performance data dictionary
    :param mem: Memory data dictionary
    :return:
    """

    for func in performance["dask_profile"].keys():
        if "functions" in mem.keys() and func in mem["functions"]:
            max_mem = mem["max_memory"][mem["functions"] == func]
            max_mem = max_mem[max_mem > 0.0]
            if max_mem.size > 0:
                performance["dask_profile"][func]["max_memory"] = numpy.mean(max_mem)
            else:
                performance["dask_profile"][func]["max_memory"] = 0.0
            min_mem = mem["min_memory"][mem["functions"] == func]
            min_mem = min_mem[min_mem > 0.0]
            if min_mem.size > 0:
                performance["dask_profile"][func]["min_memory"] = numpy.mean(min_mem)
            else:
                performance["dask_profile"][func]["min_memory"] = 0.0
        else:
            performance["dask_profile"][func]["max_memory"] = 0.0
            performance["dask_profile"][func]["min_memory"] = 0.0

    return performance
