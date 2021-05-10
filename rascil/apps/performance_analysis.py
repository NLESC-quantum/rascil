""" RASCIL performance analysis

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
import os
import pprint
import sys
import glob

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
        help="Names of json performance files to analyse: default is all json files in working directory",
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

def sort_values(xvalues, yvalues):
    """ Sort xvalues and yvalues based on xvalues
    
    :param xvalues: Iterable of xaxis values
    :param yvalues: Iterable of yaxis values
    :return:
    """
    values = zip(xvalues, yvalues)
    sorted_values = sorted(values)
    tuples = zip(*sorted_values)
    xvalues, yvalues = [list(tuple) for tuple in tuples]
    return xvalues, yvalues

def plot(xaxis, yaxes, performances, title="", normalise=True, tag=""):
    """ Plot the set of yaxes against xaxis

    :param xaxis: Name of xaxis e.g. imaging_npixel
    :param yaxes: Name of yaxes to be plotted against xaxis
    :param performance: A list of dicts containing each containing the results for one test case
    :param title: Title for plot
    :param tag: Informational tag for file name
    :return:
    """
    plt.clf()
    plt.cla()
    
    # The input values are in the cli_args dictionary
    xvalues = [performance["cli_args"][xaxis] for performance in performances]
    
    # The profile times are in the "dask_profile" dictionary
    
    if normalise:
        # Plot the time per call for each function
        for yaxis in yaxes:
            yvalues = [performance["dask_profile"][yaxis]["time"] /
                       performance["dask_profile"][yaxis]["number_calls"]
                       for performance in performances]
            sxvalues, syvalues = sort_values(xvalues, yvalues)
            log.info(f"{pprint.pformat(list(zip(sxvalues, syvalues)))}")
            plt.loglog(sxvalues, syvalues, "-", label=yaxis)
        plt.ylabel("Processing time per call (s)")
    else:
        # Plot the total time for each function
        for yaxis in yaxes:
            yvalues = [performance["dask_profile"][yaxis]["time"] for performance in performances]
            sxvalues, syvalues = sort_values(xvalues, yvalues)
            log.info(f"{pprint.pformat(list(zip(sxvalues, syvalues)))}")
            plt.loglog(sxvalues, syvalues, "-", label=yaxis)
            
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


    if args.performance_files is not None:
        performance_files = args.performance_files
    else:
        performance_files = glob.glob("*.json")
    
    log.info(f"Reading from files {pprint.pformat(performance_files)}")

    performances = [
        performance_read(performance_file)
        for performance_file in performance_files
    ]
    x_axes = list(performances[0]["cli_args"].keys())
    log.info(f"Available xaxes {pprint.pformat(x_axes)}")
    if args.x_axis not in x_axes:
        raise ValueError(f"x axis {args.x_axis} is not in file")
    
    y_axes = list(performances[0]["dask_profile"].keys())
    log.info(f"Available yaxes {pprint.pformat(y_axes)}")

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
