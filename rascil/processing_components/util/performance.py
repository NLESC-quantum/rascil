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
    """Write the current environment to JSON file

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
        if not rsexec.using_dask:
            return

        if rsexec.client is None:
            return

        if rsexec.client.cluster is None:
            return

        if rsexec.client.cluster.scheduler_info is None:
            return

        info = {
            "nworkers": len(rsexec.client.cluster.scheduler_info["workers"]),
            "scheduler": rsexec.client.cluster.scheduler_info,
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


def performance_read_memory_data(memory_file, verbose=False):
    """Get the memusage data

    :param memory_file:
    :param verbose:
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
    """Merge memory data into performance data

    Merge the memory data per function into the performance data

    :param performance: Performance data dictionary
    :param mem: Memory data dictionary
    :return:
    """

    for func in performance["dask_profile"].keys():
        if "functions" in mem.keys() and func in mem["functions"]:
            performance["dask_profile"][func]["max_memory"] = numpy.mean(
                mem["max_memory"][mem["functions"] == func]
            )
            performance["dask_profile"][func]["min_memory"] = numpy.mean(
                mem["min_memory"][mem["functions"] == func]
            )
        else:
            performance["dask_profile"][func]["max_memory"] = 0.0
            performance["dask_profile"][func]["min_memory"] = 0.0

    return performance
