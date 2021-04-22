"""Functions for monitoring performance

"""

__all__ = ['performance_store_dict']

import json

def performance_store_dict(performance_file, key, s, indent=2, mode="w"):
    """ Store dictionary in a file

    :param args: Namespace oject from argparse
    :param indent: Number of columns indent
    :return:
    """
    if performance_file is not None:
        if mode == "w":
            with open(performance_file, mode) as file:
                s=json.dumps({key:s}, indent=indent)
                file.write(s)
        elif mode == "a":
            try:
                with open(performance_file, "r") as file:
                    previous = json.load(file)
                    previous[key]=s
                with open(performance_file, "w") as file:
                    s = json.dumps(previous, indent=indent)
                    file.write(s)
            except FileNotFoundError:
                with open(performance_file, "w") as file:
                    s = json.dumps({key: s}, indent=indent)
                    file.write(s)

