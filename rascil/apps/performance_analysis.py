""" RASCIL performance analysis

"""

import argparse
import logging
import os
import pprint
import sys

import matplotlib

#matplotlib.use("Agg")

import matplotlib.pyplot as plt

from rascil.processing_components.util.performance import (
    performance_read,
)

from rascil.apps.apps_parser import (
    apps_parser_app,
)

log = logging.getLogger("rascil-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


def cli_parser():
    """Get a command line parser and populate it with arguments

    First a CLI argument parser is created. Each function call adds more arguments to the parser.

    :return: CLI parser argparse
    """

    parser = argparse.ArgumentParser(
        description="RASCIL performance analysis", fromfile_prefix_chars="@"
    )
    
    parser.add_argument(
        "--mode", type=str, default="plot", help="Processing mode: plot"
    )

    parser.add_argument(
        "--performance_files",
        type=str,
        nargs="*",
        default=None,
        help="Names of json performance files to analyse",
    )

    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Informational tag used in plot titles and file names",
    )

    parser.add_argument(
        "--x_axis",
        type=str,
        default="imaging_npixel",
        help="Name of x axis from cli_args e.g. imaging_npixel",
    )

    parser.add_argument(
        "--y_axes",
        type=str,
        nargs="*",
        default=["skymodel_predict_calibrate",
                 "skymodel_calibrate_invert",
                 "invert_ng",
                 "restore_cube",
                 "image_scatter_facets",
                 "image_gather_facets",
                 ],
        
        help="Names of values from dask_profile to plot e.g. skymodel_predict_calibrate",
    )

    return parser


def plot(xaxis, yaxes, performances, title="", normalise=True, tag=""):
    """

    :param xaxis:
    :param yaxes:
    :param performance:
    :return:
    """
    plt.clf()
    
    xvalues = [performance["cli_args"][xaxis] for performance in performances]

    if normalise:
        for yaxis in yaxes:
            yvalues = [performance["dask_profile"][yaxis]["time"] /
                       performance["dask_profile"][yaxis]["number_calls"]
                       for performance in performances]
            log.info(f"{yaxis}: {yvalues}")
            plt.loglog(xvalues, yvalues, "-", label=yaxis)
        plt.ylabel("Processing time per call (s)")
    else:
        for yaxis in yaxes:
            yvalues = [performance["dask_profile"][yaxis]["time"] for performance in performances]
            log.info(f"{yaxis}: {yvalues}")
            plt.loglog(xvalues, yvalues, "-", label=yaxis)
        clock_time = [performance["dask_profile"]["summary"]["duration"] for performance in performances]
        plt.loglog(xvalues, clock_time, "--", label="clock_time")
        processor_time = [performance["dask_profile"]["summary"]["total"] for performance in performances]
        plt.loglog(xvalues, processor_time, "--", label="processor_time")
        plt.ylabel("Total processing time (s)")

    plt.title(f"{tag} {title}")
    plt.xlabel(xaxis)
    plt.legend()
    if title is not "" or tag is not "":
        figure = f"{tag}_{title}.png"
        plt.savefig(figure)
    else:
        figure = None
        
    plt.show(block=False)
    return figure
    


def analyser(args):
    """Analyser

    The return contains names of the plot files written to disk

    :param args: argparse with appropriate arguments
    :return: Names of output plot files
    """

    # We need to tell all the Dask workers to use the same log
    cwd = os.getcwd()

    log.info("\nPerformance analyser\n")

    log.info(pprint.pformat(vars(args)))

    log.info("Current working directory is {}".format(cwd))

    performances = [
        performance_read(performance_file)
        for performance_file in args.performance_files
    ]
    x_axes = performances[0]["cli_args"].keys()
    log.info(f"Available xaxes {x_axes}")
    if args.x_axis not in x_axes:
        raise ValueError(f"x axis {args.x_axis} is not in file")
    
    y_axes = performances[0]["dask_profile"].keys()
    log.info(f"Available yaxes {y_axes}")

    for yaxis in args.y_axes:
        if yaxis not in y_axes:
            raise ValueError(f"y axis {yaxis} is not in file")
        
    tag = performances[0]["environment"]["hostname"]
    
    if args.tag is not "":
        tag = f"{args.tag}: {tag}"

    return [plot(args.x_axis, args.y_axes, performances, title="total_time",
                 normalise=False, tag=tag),
            plot(args.x_axis, args.y_axes, performances, title="time_per_call",
                 normalise=True, tag=tag)]

def main():
    # Get command line inputs
    parser = cli_parser()
    args = parser.parse_args()
    plot_files = analyser(args)
    log.info(f"Written plot files {plot_files}")


if __name__ == "__main__":
    main()
