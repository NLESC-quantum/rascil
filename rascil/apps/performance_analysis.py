""" RASCIL performance analysis

We measure the execution statistics for functions run by Dask and write to a json file.
The statistics are:

 - Total processor time per function
 - Processor time per function call
 - Number of calls per function
 - Fraction of total processor time per function
 - Maximum memory per task
 - Minimum memory per task

This app allows plotting of these statistics

- Line plots e.g. Given functions (e.g. "invert_ng") vs parameter (e.g. "imaging_npixel")
- Bar charts: statistics per function
- Contour plots e.g. Given functions (e.g. "invert_ng" vs parameters
 (e.g. "imaging_npixel", "blockvis_nvis"))
 
A typical json file looks like:

{
  "environment": {
    "git": "b'022f36072ae3cc62d79f9b507581e490ed3b35dd\\n'",
    "cwd": "/home/wangfeng/work/ska-sim-mid/continuum_imaging/resource_modelling",
    "hostname": "astrolab-hpc-1"
  },
  "cli_args": {
    "mode": "cip",
    "logfile": null,
    "performance_file": "/mnt/storage-arch/ska/ska1-mid/performance_rascil_imager_12288.json",
    "ingest_msname": "/mnt/storage-arch/ska/ska1-mid/SKA_MID_SIM_custom_B2_dec_-45.0_nominal_nchan100.ms",
    "ingest_dd": [
      0
    ],
    "ingest_vis_nchan": 100,
    "ingest_chan_per_blockvis": 16,
    "ingest_average_blockvis": "False",
    "imaging_phasecentre": null,
    "imaging_pol": "stokesI",
    "imaging_nchan": 1,
    "imaging_context": "ng",
    "imaging_ng_threads": 4,
    "imaging_w_stacking": true,
    "imaging_npixel": 12288,
    "imaging_cellsize": 5e-06,
    "imaging_weighting": "uniform",
    "imaging_robustness": 0.0,
    "imaging_dopsf": "False",
    "imaging_dft_kernel": null,
    "calibration_T_first_selfcal": 1,
    "calibration_T_phase_only": "True",
    "calibration_T_timeslice": null,
    "calibration_G_first_selfcal": 3,
    "calibration_G_phase_only": "False",
    "calibration_G_timeslice": null,
    "calibration_B_first_selfcal": 4,
    "calibration_B_phase_only": "False",
    "calibration_B_timeslice": null,
    "calibration_global_solution": "True",
    "calibration_context": "T",
    "clean_algorithm": "mmclean",
    "clean_scales": [
      0
    ],
    "clean_nmoment": 3,
    "clean_nmajor": 10,
    "clean_niter": 1000,
    "clean_psf_support": 256,
    "clean_gain": 0.1,
    "clean_threshold": 3e-05,
    "clean_component_threshold": null,
    "clean_component_method": "fit",
    "clean_fractional_threshold": 0.3,
    "clean_facets": 8,
    "clean_overlap": 32,
    "clean_taper": "tukey",
    "clean_restore_facets": 4,
    "clean_restore_overlap": 32,
    "clean_restore_taper": "tukey",
    "clean_restored_output": "list",
    "use_dask": "True",
    "dask_nthreads": null,
    "dask_memory": null,
    "dask_nworkers": null,
    "dask_scheduler": null
  },
  "restored": {
    "shape": "(6, 1, 12288, 12288)",
    "max": 0.5233755848699991,
    "min": -0.00036648297187648435,
    "maxabs": 0.5233755848699991,
    "rms": 0.000154313052487079,
    "sum": 185.61892222682283,
    "medianabs": 1.3635177033879871e-06,
    "medianabsdevmedian": 1.3635177033879871e-06,
    "median": 0.0
  },
  "residual": {
    "shape": "(6, 1, 12288, 12288)",
    "max": 0.0007164552750843033,
    "min": -0.00019836203580012325,
    "maxabs": 0.0007164552750843033,
    "rms": 2.1910916210279352e-06,
    "sum": 2.630738479114455,
    "medianabs": 1.4186849583003684e-06,
    "medianabsdevmedian": 1.4186821968807432e-06,
    "median": 9.019315938163155e-10
  },
  "deconvolved": {
    "shape": "(6, 1, 12288, 12288)",
    "max": 0.3026695593262589,
    "min": -0.0027600282599595074,
    "maxabs": 0.3026695593262589,
    "rms": 2.86755396962732e-05,
    "sum": 7.061819711500726,
    "medianabs": 0.0,
    "medianabsdevmedian": 0.0,
    "median": 0.0
  },
  "dask_profile": {
    "create_blockvisibility_from_ms": {
      "time": 503.59364795684814,
      "fraction": 0.8297172646180057,
      "number_calls": 12
    },
    "getitem": {
      "time": 0.23853826522827148,
      "fraction": 0.0003930139264760643,
      "number_calls": 23596
    },
    "convert_blockvisibility_to_stokesI": {
      "time": 43.000044107437134,
      "fraction": 0.07084656273967438,
      "number_calls": 12
    },
    ......
    "skymodel_restore_component": {
      "time": 84.2855749130249,
      "fraction": 0.13886830572093206,
      "number_calls": 12
    },
    "set_clean_beam": {
      "time": 0.02121114730834961,
      "fraction": 3.494733342149195e-05,
      "number_calls": 12
    },
    "summary": {
      "total": 60694.60880613327,
      "duration": 16539.333297729492,
      "speedup": 3.66971314463234
    }
  }
}

In addition, we can process csv files from dask-memusage. This is currently under evaluation and is
of limited help. The memusage.csv files contain the maximum and minimum memory of active tasks sampled
every 10ms. For example:

task_key,min_memory_mb,max_memory_mb
create_blockvisibility_from_ms-aafef67d-c36b-4413-80bd-90516eb920b5,87.3046875,7669.02734375
getitem-5a574568210d4ac3d72ed675dab6444a,0,0
performance_blockvisibility-9ba1c47a-f4fd-48bc-88f5-5ff4c7f39db7,0,0
create_blockvisibility_from_ms-e25eaa39-66ca-452c-9557-9f00d33bdcb4,90.32421875,7618.20703125

The mode memory_histogram will plot histograms of these values for each specified function.
The other plots will include memory information as appropriate.

This app supports plotting of various yaxes against an xaxis e.g. imaging-npixel, contours of parameter sweeps,
and barcharts
"""

import argparse
import logging
import pprint
import sys
import glob
import numpy
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from rascil.processing_components.util.performance import (
    performance_read,
    performance_read_memory_data,
)

log = logging.getLogger("rascil-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


def analyser(args):
    """Analyser of RASCIL performance files

    :param args: argparse with appropriate arguments
    :return: Names of output plot files
    """
    log.info("\nRASCIL performance analysis\n")

    verbose = args.verbose == "True"
    if verbose:
        log.info(pprint.pformat(vars(args)))

    if args.performance_files is not None:
        performance_files = args.performance_files
    else:
        performance_files = glob.glob("*.json")

    if args.mode == "line":
        # Performance data vs one parameter (e.g. blockvis_nvis)
        performances = get_performance_data(args, performance_files)
        plotfiles = plot_performance_lines(
            args.parameters[0],
            args.functions,
            performances,
            tag=args.tag,
            results=args.results,
        )
        return plotfiles
    elif args.mode == "contour":
        # Performance data vs two parameters (e.g. blockvis_nvis, imaging_npixel_sq)
        performances = get_performance_data(args, performance_files)
        plotfiles = plot_performance_contour(
            args.parameters,
            args.functions,
            performances,
            tag=args.tag,
            results=args.results,
        )
        return plotfiles

    elif args.mode == "fit":

        # Summary vs two parameters (e.g. blockvis_nvis, imaging_npixel_sq)
        performances = get_performance_data(args, performance_files)

        results = fit_summary(args.parameters, performances)

        return results

    elif args.mode == "summary":

        # Summary vs two parameters (e.g. blockvis_nvis, imaging_npixel_sq)

        performances = get_performance_data(args, performance_files)

        plotfiles = plot_summary_contour(
            args.parameters, performances, tag=args.tag, results=args.results
        )

        return plotfiles

    elif args.mode == "bar":
        # Bar chart of performance data versus function
        performances = get_performance_data(args, performance_files)
        plotfiles = plot_performance_barchart(
            performance_files, performances, tag=args.tag, results=args.results
        )
        return plotfiles
    elif args.mode == "memory_histogram":
        # Histogram of memory usage throughout one run
        memory = performance_read_memory_data(args.memory_file)
        plotfiles = plot_memory_histogram(
            args.functions, memory, tag=args.tag, results=args.results
        )
        return plotfiles
    else:
        raise ValueError(f"Unknown mode {args.mode}")


def cli_parser():
    """Get a command line parser and populate it with arguments
    First a CLI argument parser is created. Each function call adds more arguments to the parser.

    :return: CLI parser argparse
    """

    parser = argparse.ArgumentParser(
        description="RASCIL performance analysis", fromfile_prefix_chars="@"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="summary",
        help="Processing mode: line | bar | contour | summary | fit",
    )

    parser.add_argument(
        "--performance_files",
        type=str,
        nargs="*",
        default=None,
        help="Names of json performance files to analyse: default is all json files in working directory",
    )

    parser.add_argument(
        "--memory_file",
        type=str,
        default=None,
        help="Name of memusage csv file",
    )

    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Informational tag used in plot titles and file names",
    )

    parser.add_argument(
        "--parameters",
        type=str,
        nargs="*",
        default=["imaging_npixel_sq", "blockvis_nvis"],
        help="Name of parameters from cli_args e.g. imaging_npixel_sq, used for line (1 parameter)"
        " and contour plots (2 parameters)",
    )

    parser.add_argument(
        "--functions",
        type=str,
        nargs="*",
        default=[
            "skymodel_predict_calibrate",
            "skymodel_calibrate_invert",
            "invert_ng",
            "restore_cube",
            "image_scatter_facets",
            "image_gather_facets",
        ],
        help="Names of values from dask_profile to plot e.g. skymodel_predict_calibrate",
    )
    parser.add_argument(
        "--blockvis_nvis",
        type=str,
        default=None,
        help="Number of visibilities for use if blockvis_nvis not in json files",
    )

    parser.add_argument(
        "--verbose",
        type=str,
        default="False",
        help="Verbose output?",
    )

    parser.add_argument(
        "--results",
        type=str,
        default="./",
        help="Directory for results, default is current directory",
    )

    return parser


def sort_values(xvalues, yvalues):
    """Sort list xvalues and yvalues based on xvalues

    :param xvalues: Iterable of xaxis values
    :param yvalues: Iterable of yaxis values
    :return:
    """
    values = zip(xvalues, yvalues)
    sorted_values = sorted(values)
    tuples = zip(*sorted_values)
    xvalues, yvalues = [list(tuple) for tuple in tuples]
    return xvalues, yvalues


def plot_performance_lines(parameter, functions, performances, tag="", results="./"):
    """Plot the set of yaxes against xaxis

    :param parameter: Name of parameter e.g. imaging_npixel
    :param functions: Name of functions to be plotted against parameter
    :param performances: A list of dicts each containing the results for one test case
    :param tag: Informational tag for file name
    :param results: Where to place results
    :return: List of plot files
    """

    log.info("Plotting lines")

    figures = list()

    title = os.path.basename(os.getcwd())

    # The profile times are in the "dask_profile" dictionary

    has_memory = "max_memory" in performances[0]["dask_profile"]["getitem"].keys()
    if has_memory:
        axis_types = [
            (0, "Time per call (s)", "per_call"),
            (1, "Total (s)", "total"),
            (2, "Percentage time", "percentage"),
            (3, "Number calls", "number_calls"),
            (4, "Maximum memory (GB)", "maximum_memory"),
            (5, "Minimum memory (GB)", "minimum_memory"),
        ]
    else:
        axis_types = [
            (0, "Time per call (s)", "per_call"),
            (1, "Total (s)", "total"),
            (2, "Percentage time", "percentage"),
            (3, "Number calls", "number_calls"),
        ]

    xvalues = [performance["inputs"][parameter] for performance in performances]
    for axis_type, axis_type_name, axis_type_short in axis_types:
        plt.clf()
        plt.cla()

        for func in functions:
            yvalues = [
                get_data(performance, func)[axis_type] for performance in performances
            ]
            sxvalues, syvalues = sort_values(xvalues, yvalues)
            plt.loglog(sxvalues, syvalues, "-", label=func)
            plt.ylabel(axis_type_name)

        # If we are plotting total time, add the overall times
        if axis_type == 1:
            clock_time = [
                performance["dask_profile"]["summary"]["duration"]
                for performance in performances
            ]
            sxvalues, sclock_time = sort_values(xvalues, clock_time)
            plt.loglog(sxvalues, sclock_time, "--", label="clock_time")
            processor_time = [
                performance["dask_profile"]["summary"]["total"]
                for performance in performances
            ]
            sxvalues, sprocessor_time = sort_values(xvalues, processor_time)
            plt.loglog(sxvalues, sprocessor_time, "--", label="processor_time")
            plt.ylabel("Total processing time (s)")

        plt.title(f"{title} {parameter} {tag}")
        plt.xlabel(parameter)
        plt.legend()
        if title != "" or tag != "":
            if tag == "":
                figure = f"{results}/{title}_{parameter}_{axis_type_short}_line.png"
            else:
                figure = (
                    f"{results}/{title}_{parameter}_{tag}_{axis_type_short}_line.png"
                )
            plt.savefig(figure)
        else:
            figure = None

        plt.show(block=False)
        figures.append(figure)
    return figures


def plot_memory_histogram(functions, memory, tag="", results="./"):
    """Plot the memory use histograms for a set of functions

    :param functions: Name of functions to be plotted
    :param memory: Memory information
    :param tag: Informational tag for file name
    :param results: Where to place results
    :return: List of plot files
    """

    log.info("Plotting memory histograms")

    figures = list()

    title = os.path.basename(os.getcwd())

    if functions is None or functions == "" or functions == [""]:
        functions = numpy.unique(memory["functions"])

    for function in functions:
        for value_short_type, value_type in [("max_memory", "Maximum memory (GB)")]:
            mem = memory[value_short_type][memory["functions"] == function]

            plt.clf()
            plt.cla()
            plt.hist(
                mem,
            )
            plt.title(f"{function} {tag}")
            plt.xlabel(value_type)
            if title != "" or tag != "":
                if tag == "":
                    figure = (
                        f"{results}/{title}_{function}_{value_short_type}_histogram.png"
                    )
                else:
                    figure = f"{results}/{title}_{function}_{tag}_{value_short_type}_histogram.png"
                plt.savefig(figure)
            else:
                figure = None

            plt.show(block=False)
            figures.append(figure)

    return figures


def plot_summary_contour(parameters, performances, tag="", results="./"):
    """Plot the summary info as a contour

    :param parameters: Name of parameters e.g. imaging_npixel, blockvis_nvis
    :param performances: A list of dicts each containing the results for one test case
    :param tag: Informational tag for file name
    :param results: Where to place results
    :return: List of plot files
    """
    log.info("Plotting summary contours")

    figures = list()

    func = "summary"

    title = os.path.basename(os.getcwd())

    for value_type, value_type_name, value_type_short in [
        (0, "Total processor time (s)", "processor_time"),
        (1, "Duration (s)", "duration"),
        (2, "Speedup", "speedup"),
    ]:
        try:
            xvalues, yvalues, zvalues = get_summary_performance_data(
                parameters, performances, value_type
            )

            plt.clf()
            plt.cla()

            cmap = cm.get_cmap(name="tab20", lut=None)
            plt.tricontour(xvalues, yvalues, zvalues, levels=20, cmap=cmap)

            plt.title(f"{title} {tag} {value_type_name}")
            plt.xlabel(parameters[0])
            plt.ylabel(parameters[1])
            plt.colorbar()
            if title != "" or tag != "":
                if tag == "":
                    figure = f"{results}/{title}_{func}_{value_type_short}_contour.png"
                else:
                    figure = (
                        f"{results}/{title}_{func}_{tag}_{value_type_short}_contour.png"
                    )
                plt.savefig(figure)
                figures.append(figure)

            plt.show(block=False)
        except RuntimeError as e:
            # Most common cause is that the axes are degenerate e.g. all same value on x axis
            log.error(f"RuntimeError: {value_type_short}, {func}, {e}")

    return figures


def fit_2d_plane(x, y, z) -> (float, float):
    """Fit the best fitting plane p x + q y = z
    :return: parameters p, q defining plane
    """
    sx2 = numpy.sum(x * x)
    sy2 = numpy.sum(y * y)
    sxy = numpy.sum(x * y)
    sxz = numpy.sum(x * z)
    syz = numpy.sum(y * z)
    mat = numpy.matrix([[sx2, sxy], [sxy, sy2]])
    mat = numpy.linalg.inv(mat)
    pq = numpy.matmul(mat, [sxz, syz])
    return pq[0, 0], pq[0, 1]


def fit_summary(parameters, performances):
    """Fit the summary data by a 2D plane.

    :param parameters: Name of parameters e.g. imaging_npixel, blockvis_nvis
    :param performances: A list of dicts each containing the results for one test case
    :returns: Dictionary containing fit information
    """
    log.info("Fitting planes to summary values")

    fits = {}

    for value_type, value_type_name, value_type_short in [
        (0, "Total processor time (s)", "processor_time"),
        (1, "Duration (s)", "duration"),
        (2, "Speedup", "speedup"),
    ]:
        xvalues, yvalues, zvalues = get_summary_performance_data(
            parameters, performances, value_type
        )
        p, q = fit_2d_plane(xvalues, yvalues, zvalues)

        log.info(
            f"{value_type_name} =\n\t {p:g} * {parameters[0]} + {q:g} * {parameters[1]}"
        )
        fits[value_type_short] = {"parameters": parameters[0:2], "p": p, "q": q}

    return fits


def plot_performance_contour(parameters, functions, performances, tag="", results="./"):
    """Plot the performance value against two parameters for all axis types

    :param parameters: Name of parameters e.g. imaging_npixel, blockvis_nvis
    :param functions: Name of functions to be plotted against parameter
    :param performances: A list of dicts each containing the results for one test case
    :param tag: Informational tag for file name
    :param results: Where to place results
    :return: List of plot files
    """
    log.info("Plotting contours")

    figures = list()

    title = os.path.basename(os.getcwd())

    has_memory = "max_memory" in performances[0]["dask_profile"]["getitem"].keys()
    if has_memory:
        axis_types = [
            (0, "Time per call (s)", "per_call"),
            (1, "Total (s)", "total"),
            (2, "Percentage time", "percentage"),
            (3, "Number calls", "number_calls"),
            (4, "Maximum memory (GB)", "maximum_memory"),
            (5, "Minimum memory (GB)", "minimum_memory"),
        ]
    else:
        axis_types = [
            (0, "Time per call (s)", "per_call"),
            (1, "Total (s)", "total"),
            (2, "Percentage time", "percentage"),
            (3, "Number calls", "number_calls"),
        ]

    for func in functions:

        for value_type, value_type_name, value_type_short in axis_types:
            xvalues, yvalues, zvalues = get_performance_contour_data(
                func, parameters, performances, value_type
            )

            try:
                if (
                    xvalues is not None
                    and yvalues is not None
                    and zvalues is not None
                    and numpy.max(zvalues) > numpy.min(zvalues)
                ):
                    plt.clf()
                    plt.cla()

                    cmap = cm.get_cmap(name="tab20", lut=None)
                    plt.tricontour(xvalues, yvalues, zvalues, levels=20, cmap=cmap)

                    plt.title(f"{title} {func} {tag} {value_type_name}")
                    plt.xlabel(parameters[0])
                    plt.ylabel(parameters[1])
                    plt.colorbar()
                    if title != "" or tag != "":
                        if tag == "":
                            figure = f"{results}/{title}_{func}_{value_type_short}_contour.png"
                        else:
                            figure = f"{results}/{title}_{func}_{tag}_{value_type_short}_contour.png"
                        plt.savefig(figure)
                        figures.append(figure)

                    plt.show(block=False)
            except RuntimeError as e:
                # Probably the axes are singular i.e. one axis is all the same value
                log.error(f"RuntimeError: {value_type_short}, {func}, {e}")

    return figures


def get_performance_contour_data(func, parameters, performances, axis_type):
    """Get the performance info for given axis type for use in contour plots

    :param func: Name of function e.g. "invert_ng"
    :param parameters: Name of parameters e.g. "imaging_npixel", "blockvis_nvis"
    :param performances: Performance information
    :param axis_type: Code: 0=per call, 1=total, 2=fraction, 3=max_memory, 4=min_memory
    :return: xvalues, yvalues, zvalues
    """
    xvalues = numpy.array(
        [performance["inputs"][parameters[0]] for performance in performances]
    )
    yvalues = numpy.array(
        [performance["inputs"][parameters[1]] for performance in performances]
    )
    zvalues = numpy.array(
        [get_data(performance, func)[axis_type] for performance in performances]
    )
    return xvalues, yvalues, zvalues


def get_summary_performance_data(parameters, performances, axis_type):
    """Get the summary for given axis type

    :param parameters: Name of parameters e.g. "imaging_npixel", "blockvis_nvis"
    :param performances: Performance information
    :param axis_type: Code: 0=total, 1=duration, 2=speedup
    :return: x, y, z
    """
    xvalues = numpy.array(
        [performance["inputs"][parameters[0]] for performance in performances]
    )
    yvalues = numpy.array(
        [performance["inputs"][parameters[1]] for performance in performances]
    )
    zvalues = numpy.array(
        [get_summary_data(performance)[axis_type] for performance in performances]
    )
    return xvalues, yvalues, zvalues


def get_data_sizes(performance):
    """Get sizes of the visibility and final image

    :param performance: Dictionary containing performance information for one run
    :return:
    """
    imagesize = performance["restored"]["size"] * 2**-30
    nblockvis = (
        performance["inputs"]["ingest_vis_nchan"]
        // performance["inputs"]["ingest_chan_per_blockvis"]
    )
    vissize = nblockvis * performance["blockvis0"]["size"] * 2**-30
    return imagesize, vissize


def plot_performance_barchart(
    performance_files, performances, tag="", results="./", plot_sizes=True
):
    """Plot barchart of the six types of performance data for all functions

    :param performance_files: Names of performance files
    :param performances: Performance info for each performance file
    :param results: Where to place results
    :param plot_sizes: Plot the total visibility size (GB) and final image size (GB)
    :param tag: Informative tag for file name
    :return: Names of plot files
    """
    figures = list()

    log.info("Plotting barcharts")

    for ipf, performance_file in enumerate(performance_files):
        performance = performances[ipf]
        (
            time_per_call,
            total_time,
            fraction_time,
            number_calls,
            max_memory,
            min_memory,
            functions,
        ) = get_performance_barchart_data(performance)

        image_size, vis_size = get_data_sizes(performance)

        title = os.path.basename(performance_file.replace(".json", ""))
        title = title.replace(
            "performance_rascil_imager", os.path.basename(os.getcwd())
        )

        # The profile times are in the "dask_profile" dictionary
        for axis, axis_type, axis_type_short in [
            (total_time, "Total (s)", "total"),
            (time_per_call, "Per call (s)", "per_call"),
            (fraction_time, "Percentage", "percentage"),
            (number_calls, "Number calls", "number_calls"),
            (max_memory, "Maximum memory (GB)", "maximum_memory"),
            (min_memory, "Minimum memory (GB)", "minimum_memory"),
        ]:
            if axis is not None:
                plt.clf()
                plt.cla()
                y_pos = numpy.arange(len(axis))
                saxis, _ = sort_values(axis, axis)
                _, syaxes = sort_values(axis, functions)
                plt.barh(y_pos, saxis, align="center", alpha=0.5)
                plt.yticks(y_pos, syaxes, fontsize="x-small")
                plt.xlabel(axis_type)
                plt.title(f"{title} {tag} {axis_type}")
                if plot_sizes and axis_type_short in [
                    "maximum_memory",
                    "minimum_memory",
                ]:
                    plt.axvline(
                        x=image_size,
                        label="final image size",
                        color="r",
                        linestyle="--",
                    )
                    plt.axvline(
                        x=vis_size, label="total vis size", color="b", linestyle="--"
                    )
                    plt.legend()

                plt.tight_layout()
                plt.show(block=False)
                if title != "" or tag != "":
                    if tag == "":
                        figure = f"{results}/{title}_{axis_type_short}_bar.png"
                    else:
                        figure = f"{results}{title}_{tag}_{axis_type_short}_bar.png"

                    plt.savefig(figure)
                    figures.append(figure)

    return figures


def get_performance_barchart_data(performance):
    """Get performance data for barchart

    :param performance: Performance dictionary associated with one file
    :return: time_per_call, total_time, fraction_time, number_calls, max_memory,
        min_memory, functions,

    """
    # The input values are in the inputs dictionary
    functions = list()
    for key in performance["dask_profile"].keys():
        if key != "summary":
            functions.append(key)

    total_time = [performance["dask_profile"][func]["time"] for func in functions]

    fraction_time = [
        performance["dask_profile"][func]["fraction"] for func in functions
    ]

    time_per_call = [
        performance["dask_profile"][func]["time"]
        / performance["dask_profile"][func]["number_calls"]
        for func in functions
    ]
    number_calls = [
        performance["dask_profile"][func]["number_calls"] for func in functions
    ]

    try:
        max_memory = [
            performance["dask_profile"][func]["max_memory"] for func in functions
        ]

        min_memory = [
            performance["dask_profile"][func]["min_memory"] for func in functions
        ]
    except KeyError:
        max_memory = None
        min_memory = None

    return (
        time_per_call,
        total_time,
        fraction_time,
        number_calls,
        max_memory,
        min_memory,
        functions,
    )


def get_data(performance, func):
    """Get the dask_profile performance data for a given function

    :param performance: Single performance dict
    :param func: Name of function
    :return: time_per_call, total_time, fraction_time, number_calls, max_memory, min_memory
    """
    total_time = performance["dask_profile"][func]["time"]
    time_per_call = (
        performance["dask_profile"][func]["time"]
        / performance["dask_profile"][func]["number_calls"]
    )
    fraction_time = performance["dask_profile"][func]["fraction"]
    number_calls = performance["dask_profile"][func]["number_calls"]

    if "max_memory" in performance["dask_profile"][func].keys():
        max_memory = performance["dask_profile"][func]["max_memory"]
    else:
        max_memory = None

    if "min_memory" in performance["dask_profile"][func].keys():
        min_memory = performance["dask_profile"][func]["min_memory"]
    else:
        min_memory = None

    return (
        time_per_call,
        total_time,
        fraction_time,
        number_calls,
        max_memory,
        min_memory,
    )


def get_summary_data(performance):
    """Get the dask_profile summary data

    :param performance: Single performance dict
    :return: total_time, duration, speedup
    """
    total_time = performance["dask_profile"]["summary"]["total"]
    duration = performance["dask_profile"]["summary"]["duration"]
    speedup = performance["dask_profile"]["summary"]["speedup"]
    return (
        total_time,
        duration,
        speedup,
    )


def get_performance_data(args, performance_files, verbose=False):
    """Read and normalize performance data

    :param args: CLI args
    :param performance_files: Names of performance files
    :return: performances dict
    """

    if verbose:
        log.info(f"Reading from files {pprint.pformat(performance_files)}")

    performances = [
        performance_read(performance_file) for performance_file in performance_files
    ]
    if performances is None or len(performances) == 0:
        raise ValueError(f"Unable to read performance data {performance_files}")
    # inputs are made of cli_args and blockvis information
    for perf in performances:
        perf["inputs"] = perf["cli_args"]
        if "blockvis0" in perf.keys():
            perf["inputs"]["blockvis_nvis"] = (
                perf["blockvis0"]["number_times"]
                * perf["blockvis0"]["number_baselines"]
                * perf["blockvis0"]["nchan"]
                * perf["blockvis0"]["npol"]
            )
        else:
            perf["inputs"]["blockvis_nvis"] = args.blockvis_nvis

        perf["inputs"] = {
            **perf["inputs"],
            "imaging_npixel_sq": perf["inputs"]["imaging_npixel"] ** 2,
        }

    parameters = list(performances[0]["inputs"].keys())
    functions = list(performances[0]["dask_profile"].keys())
    if verbose:
        log.info(f"Available parameters {pprint.pformat(parameters)}")
        log.info(f"Available functions {pprint.pformat(functions)}")
    for func in args.functions:
        if func not in functions:
            raise ValueError(f"Function {func} is not in file")
    return performances


def main():
    # Get command line inputs
    parser = cli_parser()
    args = parser.parse_args()
    plot_files = analyser(args)
    log.info("Results:")
    log.info(pprint.pformat(plot_files))


if __name__ == "__main__":
    main()
