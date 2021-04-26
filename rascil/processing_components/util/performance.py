"""Functions for monitoring performance

"""

__all__ = [
    "performance_store_dict",
    "performance_qa_image",
    "performance_dask_configuration",
    "git_hash",
]

import json
import logging
import os
import sys
import socket

from rascil.processing_components.image.operations import qa_image

log = logging.getLogger("rascil-logger")


def git_hash():
    """Get the hash for this git repository.

    Requires that the code tree was created using git

    :return: string or "unknown"
    """
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"])
    except Exception as excp:
        log.info(excp)
        return "unknown"


def performance_environment(performance_file, indent=2, mode="a"):
    """Write th current environment to JSON file

    :param performance_file:
    :param indent:
    :param mode:
    :return:
    """
    info = {
        "script": sys.argv[0],
        "git": str(git_hash()),
        "cwd": os.getcwd(),
        "hostname": socket.gethostname(),
    }

    performance_store_dict(
        performance_file, "environment", info, indent=indent, mode=mode
    )


def performance_dask_configuration(performance_file, indent=2, mode="a"):
    """Write Dask configuration info to performance file

    :param performance_file:
    :param key:
    :param indent:
    :param mode:
    :return:
    """
    from rascil.workflows.rsexecute.execution_support import rsexecute

    if not rsexecute.using_dask:
        return
    
    if rsexecute.client is None:
        return

    if rsexecute.client.cluster is None:
        return

    if rsexecute.client.cluster.scheduler_info is None:
        return

    info = {
        "nworkers": len(rsexecute.client.cluster.scheduler_info["workers"]),
        "scheduler": rsexecute.client.cluster.scheduler_info,
    }
    performance_store_dict(
        performance_file, "dask_configuration", info, indent=indent, mode=mode
    )


def performance_qa_image(performance_file, key, im, indent=2, mode="a"):
    """Store image qa in a performance file

    :param key: Key for s for be stored as e.g. "restored"
    :param im: Image
    :param indent: Number of columns indent
    :return:
    """

    qa = qa_image(im)
    performance_store_dict(performance_file, key, qa.data, indent=indent, mode=mode)


def performance_store_dict(performance_file, key, s, indent=2, mode="a"):
    """Store dictionary in a file using json

    :param key: Key for s for be stored as e.g. "cli_args"
    :param s: Dictionary
    :param indent: Number of columns indent
    :return:
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
