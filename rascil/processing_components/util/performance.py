"""Functions for monitoring performance

"""

__all__ = ['performance_store_dict', 'performance_qa_image']

import json

from rascil.processing_components.image.operations import qa_image

def performance_qa_image(performance_file, key, im, indent=2, mode="a"):
    """ Store image qa in a performance file

    :param key: Key for s for be stored as e.g. "resroeed"
    :param im: Image
    :param indent: Number of columns indent
    :return:
    """

    qa = qa_image(im)
    performance_store_dict(performance_file, key, qa.data, indent=indent, mode=mode)


def performance_store_dict(performance_file, key, s, indent=2, mode="a"):
    """ Store dictionary in a file using json

    :param key: Key for s for be stored as e.g. "cli_args"
    :param s: Dictionary
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

