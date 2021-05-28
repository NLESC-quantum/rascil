""" RASCIL performance analysis

We measure the execution statistics for functions run by Dask and write to a json file.
The statistics are:

 - Total processor time per function
 - Processor time per function call
 - Number of calls per function
 - Fraction of total processor time per function

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
    "create_image_from_visibility": {
      "time": 360.859938621521,
      "fraction": 0.5945502338999434,
      "number_calls": 12
    },
    "imaging_grid_weights": {
      "time": 500.3804063796997,
      "fraction": 0.8244231509556011,
      "number_calls": 12
    },
    "griddata_merge_weights": {
      "time": 18.455530881881714,
      "fraction": 0.030407199658920542,
      "number_calls": 2
    },
    "imaging_re_weight": {
      "time": 154.4264853000641,
      "fraction": 0.2544319641194542,
      "number_calls": 12
    },
    "SkyModel": {
      "time": 0.002610445022583008,
      "fraction": 4.300950403883679e-06,
      "number_calls": 12
    },
    "imaging_zero_vis": {
      "time": 10.72262978553772,
      "fraction": 0.017666527549073163,
      "number_calls": 12
    },
    "skymodel_predict_calibrate": {
      "time": 10259.664942264557,
      "fraction": 16.903750010211457,
      "number_calls": 144
    },
    "imaging_subtract_vis": {
      "time": 451.71227264404297,
      "fraction": 0.7442378845984041,
      "number_calls": 144
    },
    "skymodel_calibrate_invert": {
      "time": 25574.409220695496,
      "fraction": 42.13621229915756,
      "number_calls": 144
    },
    "skymodel_update_components": {
      "time": 0.0013587474822998047,
      "fraction": 2.2386625583827824e-06,
      "number_calls": 12
    },
    "getattr": {
      "time": 0.002603292465209961,
      "fraction": 4.289165901909388e-06,
      "number_calls": 132
    },
    "invert_ng": {
      "time": 2501.930527448654,
      "fraction": 4.122162703841055,
      "number_calls": 12
    },
    "sum_invert_results": {
      "time": 148.35386657714844,
      "fraction": 0.2444267612812377,
      "number_calls": 14
    },
    "imaging_extract_psf": {
      "time": 297.21881437301636,
      "fraction": 0.48969557629471366,
      "number_calls": 120
    },
    "concat_images": {
      "time": 6363.035717487335,
      "fraction": 10.483691785232072,
      "number_calls": 21900
    },
    "image_scatter_facets": {
      "time": 7640.010337114334,
      "fraction": 12.587625964470012,
      "number_calls": 264
    },
    "normalize_sumwt": {
      "time": 1.0463624000549316,
      "fraction": 0.00172397914845642,
      "number_calls": 2
    },
    "fit_psf": {
      "time": 0.06091737747192383,
      "fraction": 0.00010036703204810515,
      "number_calls": 2
    },
    "threshold_list": {
      "time": 42.18509531021118,
      "fraction": 0.06950385897527743,
      "number_calls": 20
    },
    "imaging_deconvolve": {
      "time": 3943.7179369926453,
      "fraction": 6.497641247823855,
      "number_calls": 1280
    },
    "image_scatter_channels": {
      "time": 1.8637990951538086,
      "fraction": 0.0030707819554567574,
      "number_calls": 1280
    },
    "image_gather_facets": {
      "time": 392.34232544898987,
      "fraction": 0.6464203875210465,
      "number_calls": 132
    },
    "skymodel_update_image": {
      "time": 669.1622524261475,
      "fraction": 1.1025069039715563,
      "number_calls": 120
    },
    "list": {
      "time": 113.77707552909851,
      "fraction": 0.1874582895698657,
      "number_calls": 8
    },
    "remove_sumwt": {
      "time": 25.487622261047363,
      "fraction": 0.04199322272997632,
      "number_calls": 2
    },
    "restore_cube": {
      "time": 592.6391408443451,
      "fraction": 0.9764279768855815,
      "number_calls": 192
    },
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

This app supports plotting and fitting of various yaxes against an xaxis e.g. imaging-npixel
"""

import argparse
import logging
import pprint
import sys
import glob
import numpy

import matplotlib.pyplot as plt

from rascil.processing_components.util.performance import performance_read_memory_data
from rascil.processing_components.util.performance import (
    performance_read,
)

log = logging.getLogger("rascil-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


def analyser(args):
    """Analyser

    The return contains names of the plot files written to disk

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
        performances = get_performance_data(args, performance_files)
        plotfiles = plot_performance_lines(
            args.parameters[0],
            args.functions,
            performances,
            tag=args.tag,
            verbose=verbose,
            results=args.results,
        )
        return plotfiles
    elif args.mode == "contour":
        performances = get_performance_data(args, performance_files)
        plotfiles = plot_performance_contour(
            args.parameters,
            args.functions,
            performances,
            tag=args.tag,
            verbose=verbose,
            results=args.results,
        )
        return plotfiles

    elif args.mode == "bar":
        performances = get_performance_data(args, performance_files)
        plotfiles = plot_performance_barchart(
            performance_files,
            performances,
            tag=args.tag,
            verbose=verbose,
            results=args.results,
        )
        return plotfiles
    elif args.mode == "memory_histogram":
        memory = performance_read_memory_data(args.memory_file)
        plotfiles = plot_memory_histogram(
            args.functions, memory, tag=args.tag, verbose=verbose, results=args.results
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
        "--mode", type=str, default="line", help="Processing mode: line | bar | contour"
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
        help="Names of memusage csv files",
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


def plot_performance_lines(
    parameter, functions, performances, title="", tag="", verbose=False, results="./"
):
    """Plot the set of yaxes against xaxis

    :param parameter: Name of parameter e.g. imaging_npixel
    :param functions: Name of functions to be plotted against parameter
    :param performance: A list of dicts containing each containing the results for one test case
    :param title: Title for plot
    :param tag: Informational tag for file name
    :return:
    """

    log.info("Plotting lines")

    figures = list()

    if title == "":
        title = "performance"

    # The profile times are in the "dask_profile" dictionary

    xvalues = [performance["inputs"][parameter] for performance in performances]
    for time_type, time_type_name, time_type_short in [
        (0, "Time per call (s)", "per_call"),
        (1, "Total (s)", "total"),
        (2, "Percentage time", "percentage"),
        (3, "Number calls", "number_calls"),
    ]:
        plt.clf()
        plt.cla()

        for func in functions:
            yvalues = [
                get_data(performance, func)[time_type] for performance in performances
            ]
            sxvalues, syvalues = sort_values(xvalues, yvalues)
            if verbose:
                log.info(f"{pprint.pformat(list(zip(sxvalues, syvalues)))}")
            plt.loglog(sxvalues, syvalues, "-", label=func)
            plt.ylabel(time_type_name)

        # If we are plotting total time, add the overall times
        if time_type == 1:
            clock_time = [
                performance["dask_profile"]["summary"]["duration"]
                for performance in performances
            ]
            plt.loglog(xvalues, clock_time, "--", label="clock_time")
            processor_time = [
                performance["dask_profile"]["summary"]["total"]
                for performance in performances
            ]
            plt.loglog(xvalues, processor_time, "--", label="processor_time")
            plt.ylabel("Total processing time (s)")

        plt.title(f"{title} {tag} {time_type_name}")
        plt.xlabel(parameter)
        plt.legend()
        if title is not "" or tag is not "":
            if tag == "":
                figure = f"{results}/{title}_{time_type_short}_line.png"
            else:
                figure = f"{results}/{title}_{tag}_{time_type_short}_line.png"
            plt.savefig(figure)
        else:
            figure = None

        plt.show(block=False)
        figures.append(figure)
    return figures


def plot_memory_histogram(
    functions, memory, title="", tag="", verbose=False, results="./"
):
    """Plot the memory use histograms for a set of functions

    :param functions: Name of functions to be plotted
    :param memory:
    :param title: Title for plot
    :param tag: Informational tag for file name
    :return:
    """

    log.info("Plotting memory histograms")

    figures = list()

    if title == "":
        title = "memory"

    if functions is None or functions == "" or functions == [""]:
        functions = numpy.unique(memory["functions"])

    for function in functions:
        for short_type, type in [("max_memory", "Maximum memory (GB)")]:
            mem = memory[short_type][memory["functions"] == function] * 2 ** -30

            plt.clf()
            plt.cla()
            plt.hist(
                mem,
            )
            plt.title(f"{function} {tag} {type}")
            plt.xlabel(type)
            if title is not "" or tag is not "":
                if tag == "":
                    figure = f"{results}/{title}_{short_type}_histogram.png"
                else:
                    figure = f"{results}/{title}_{tag}_{short_type}_histogram.png"
                plt.savefig(figure)
            else:
                figure = None

            plt.show(block=False)
            figures.append(figure)

    return figures


def plot_performance_contour(
    parameters, functions, performances, title="", tag="", verbose=False, results="./"
):
    """Plot the set of yaxes against xaxis

    :param parameters: Name of parameters e.g. imaging_npixel, blockvis_nvis
    :param functions: Name of functions to be plotted against parameter
    :param performance: A list of dicts containing each containing the results for one test case
    :param title: Title for plot
    :param tag: Informational tag for file name
    :return:
    """
    log.info("Plotting contours")

    figures = list()

    if title == "":
        title = "performance"

    for func in functions:

        for time_type, time_type_name, time_type_short in [
            (0, "Time per call (s)", "per_call"),
            (1, "Total (s)", "total"),
            (2, "Percentage time", "percentage"),
            (3, "Number calls", "number_calls"),
        ]:
            plt.clf()
            plt.cla()

            xvalues, yvalues, zvalues = get_performance_contour_data(
                func, parameters, performances, time_type
            )

            plt.tricontour(xvalues, yvalues, zvalues, levels=10)

            plt.title(f"{title} {func} {tag} {time_type_name}")
            plt.xlabel(parameters[0])
            plt.ylabel(parameters[1])
            plt.colorbar()
            if title is not "" or tag is not "":
                if tag == "":
                    figure = f"{results}/{title}_{func}_{time_type_short}_contour.png"
                else:
                    figure = (
                        f"{results}/{title}_{func}_{tag}_{time_type_short}_contour.png"
                    )
                plt.savefig(figure)
                figures.append(figure)

            plt.show(block=False)
    return figures


def get_performance_contour_data(func, parameters, performances, axis_type):
    """Get the surface data for given parameters and function

    :param func: Name of function e.g. "invert_ng"
    :param parameters: Name of parameters e.g. "imaging_npixel", "blockvis_nvis"
    :param performances: Performance information
    :param axis_type: Code: 0=per call, 1=total, 2=fraction, 3=max_memory, 4=min_memory
    :return: x, y, z
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


def plot_performance_barchart(
    performance_files, performances, title="", tag="", verbose=False, results="./"
):
    """Plot the set of yaxes

    :param performance: A list of dicts each containing the results for one test case
    :param title: Title for plot
    :param tag: Informative tag for file name
    :return:
    """
    figures = list()

    if title == "":
        title = "performance"

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

        # The profile times are in the "dask_profile" dictionary
        for axis, axis_type, axis_type_short in [
            (total_time, "Total (s)", "total"),
            (time_per_call, "Per call (s)", "per_call"),
            (fraction_time, "Percentage", "percentage"),
            (number_calls, "Number calls", "number_calls"),
            (max_memory, "Maximum memory (GB)", "maximum_memory"),
            (min_memory, "Minimum memory (GB)", "minimum_memory"),
        ]:
            plt.clf()
            plt.cla()
            y_pos = numpy.arange(len(axis))
            saxis, _ = sort_values(axis, axis)
            _, syaxes = sort_values(axis, functions)
            plt.barh(y_pos, saxis, align="center", alpha=0.5)
            plt.yticks(y_pos, syaxes, fontsize="x-small")
            plt.xlabel(axis_type)
            plt.title(f"{title} {tag} {axis_type}")
            plt.tight_layout()
            plt.show(block=False)
            if title is not "" or tag is not "":
                if tag == "":
                    figure = f"{results}/{title}_{axis_type_short}_bar.png"
                else:
                    figure = f"{results}{title}_{tag}_{axis_type_short}_bar.png"

                plt.savefig(figure)
                figures.append(figure)

    return figures


def get_performance_barchart_data(performance):
    """Get the total time, time per call, fractional time, number_calls, and allowed yaxes

    :param performance: Performance dictionary associated with one file
    :return: time_per_call, total_time, fraction_time, number_calls, functions
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

    max_memory = [performance["dask_profile"][func]["max_memory"] for func in functions]

    min_memory = [performance["dask_profile"][func]["min_memory"] for func in functions]

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
    """Get the performance data for a given function

    :param performance: Single performance dict
    :param func: Name of function
    :return: time_per_call, total_time, fraction_time, number_calls
    """
    total_time = performance["dask_profile"][func]["time"]
    time_per_call = (
        performance["dask_profile"][func]["time"]
        / performance["dask_profile"][func]["number_calls"]
    )
    fraction_time = performance["dask_profile"][func]["fraction"]
    number_calls = performance["dask_profile"][func]["number_calls"]

    return time_per_call, total_time, fraction_time, number_calls


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
    log.info("Plot files written:")
    log.info(pprint.pformat(plot_files))


if __name__ == "__main__":
    main()
