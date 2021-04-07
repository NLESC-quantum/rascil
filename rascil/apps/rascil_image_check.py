""" RASCIL app for checking statistics of an image. This is designed for use in a shell script.

"""

import argparse
import logging
import sys

from rascil.processing_components import (
    import_image_from_fits,
    qa_image
)

log = logging.getLogger("rascil-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


def cli_parser():
    """Get a command line parser and populate it with arguments

    First a CLI argument parser is created. Each function call adds more arguments to the parser.

    :return: CLI parser argparse
    """

    parser = argparse.ArgumentParser(description="RASCIL image check",
                                     fromfile_prefix_chars="@")
    parser.add_argument(
        "--image", type=str, default=None, help="Image to be read"
    )
    parser.add_argument(
        "--stat",
        type=str,
        default="max",
        help="Image QA field to check",
    )
    parser.add_argument(
        "--min",
        type=float,
        default=None,
        help="Minimum value",
    )
    parser.add_argument(
        "--max",
        type=float,
        default=None,
        help="Maximum value",
    )
    return parser


def image_check(args):
    """Provides a check on a named statistic of an image
    
    The args are:
    
    --image Image to be read
    --stat Image QA field to check
    --min Minimum value
    --max Maximum value
    
    :param args: argparse with appropriate arguments
    :return: None
    """

    if args.image is None:
        raise ValueError("Image name must be specified by --image")
    if args.max is None:
        raise ValueError(f"Maximum of image statistic {args.stat} must be specified by --max")
    if args.min is None:
        raise ValueError(f"Minimum of image statistic {args.stat} must be specified by --min")

    im = import_image_from_fits(args.image)
    qa = qa_image(im)
    
    if args.stat in qa.data.keys():
        if qa.data[args.stat] >= args.min and qa.data[args.stat] <= args.max:
            return 0
        else:
            return 1
    else:
        raise ValueError(f"{args.stat} is not a valid field from qa")

if __name__ == "__main__":
    parser = cli_parser()
    args = parser.parse_args()
    print(image_check(args))
