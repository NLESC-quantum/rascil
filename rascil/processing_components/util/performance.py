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

import json
import logging
import os
import socket
import subprocess
from datetime import datetime

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
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"performance file {performance_file} does not exist")


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
    :param key: Key to use for the configuration info e.g. "dask_configuration"
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
