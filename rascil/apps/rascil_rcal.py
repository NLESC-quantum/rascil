""" RASCIL MS advice

"""

import sys
import argparse
import logging
import pprint
from typing import Iterable

import numpy
import xarray

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.time import Time
from astropy.visualization import time_support

from rascil.data_models import (
    BlockVisibility,
    GainTable,
    export_gaintable_to_hdf5,
    import_skycomponent_from_hdf5,
)
from rascil.processing_components import (
    create_blockvisibility_from_ms,
    solve_gaintable,
    dft_skycomponent_visibility,
    copy_visibility,
    copy_gaintable,
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
        description="RASCIL RCAL simulator", fromfile_prefix_chars="@"
    )
    parser.add_argument(
        "--ingest_msname", type=str, default=None, help="MeasurementSet to be read"
    )
    parser.add_argument(
        "--ingest_dd",
        type=int,
        nargs="*",
        default=[0],
        help="Data descriptors in MS to read (all must have the same number of channels)",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default=None,
        help="Name of logfile (default is to construct one from msname)",
    )

    parser.add_argument(
        "--ingest_components_file",
        type=str,
        default=None,
        help="Name of components file (HDF5) format",
    )
    parser.add_argument(
        "--do_plotting",
        type=str,
        default=False,
        help="If yes, plot the gain table values over time",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=None,
        help="Directory to save the gain plots into (default is the same directory the MS file is located)",
    )
    parser.add_argument(
        "--use_previous_gaintable",
        type=str,
        default="True",
        help="Use previous gaintable as starting point for solution",
    )

    parser.add_argument(
        "--phase_only_solution",
        type=str,
        default="True",
        help="Solution should be for phases only",
    )

    parser.add_argument(
        "--solution_tolerance",
        type=float,
        default=1e-12,
        help="Tolerance for solution: stops iteration when changes below this level",
    )

    return parser


def rcal_simulator(args):
    """RCAL simulator

    Delivers imaging advice for an MS

    :param args: argparse with appropriate arguments
    :return: None
    """

    assert args.ingest_msname is not None, "Input msname must be specified"

    if args.logfile is None:
        logfile = args.ingest_msname.replace(".ms", "_rcal.log")
    else:
        logfile = args.logfile

    def init_logging():
        logging.basicConfig(
            filename=logfile,
            filemode="a",
            format="%(asctime)s.%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%d/%m/%Y %I:%M:%S %p",
            level=logging.INFO,
        )

    init_logging()

    log.info("\nRASCIL RCAL simulator\n")

    log.info(pprint.pformat(vars(args)))

    bvis = create_blockvisibility_from_ms(
        args.ingest_msname, selected_dds=args.ingest_dd
    )[0]

    if args.ingest_components_file is not None:
        model_components = import_skycomponent_from_hdf5(args.ingest_components_file)
    else:
        model_components = None

    if args.plot_dir is None:
        plot_dir = os.getcwd() + "plots/"
    else:
        plot_dir = args.plot_dir

    log.info(f"Write plots into directory: \n{plot_dir}\n")

    log.info(f"\nMS loaded into BlockVisibility:\n{bvis}\n")

    bvis_gen = bvis_source(bvis)

    gt_gen = bvis_solver(
        bvis_gen,
        model_components,
        phase_only=args.phase_only_solution == "True",
        use_previous=args.use_previous_gaintable == "True",
        tol=args.solution_tolerance,
    )
    full_gt = gt_sink(gt_gen, args.do_plotting, plot_dir)

    gtfile = args.ingest_msname.replace(".ms", "_gaintable.hdf")
    export_gaintable_to_hdf5(full_gt, gtfile)
    return gtfile


def bvis_source(bvis: BlockVisibility, dim="time") -> Iterable[BlockVisibility]:
    """Creates a BlockVisibility generator that iterates in the specified dimension

    :param bvis: BlockVisibility
    :param dim: dimension to be iterated
    :return: generator of BlockVisibility
    """
    log.info("\nrascil_rcal_bvis_source: BlockVisibility source starting")
    for time, bv in bvis.groupby(dim, squeeze=False):
        yield bv
    log.info("rascil_rcal_bvis_source: BlockVisibility source has finished\n")


def bvis_solver(
    bvis_gen: Iterable[BlockVisibility], model_components, use_previous=True, **kwargs
) -> Iterable[GainTable]:
    """Iterate through the block vis, solving for the gain, returning gaintable generator

    Optionally takes a list of skycomponents to use as a model

    :param bvis_gen: Generator of BlockVisibility
    :param model_components: Model components
    :param kwargs: Optional keywords
    :return: generator of GainTables
    """
    previous = None
    for bv in bvis_gen:
        if model_components is not None:
            modelvis = copy_visibility(bv)
            modelvis = dft_skycomponent_visibility(modelvis, model_components)
            gt = solve_gaintable(bv, modelvis=modelvis, gt=previous, **kwargs)
            if use_previous:
                copy_gaintable(gt, previous)
        else:
            gt = solve_gaintable(bv, gt=previous, **kwargs)
        if use_previous:
            copy_gaintable(gt, previous)
        yield gt


def gt_sink(gt_gen: Iterable[GainTable], do_plotting, plot_dir):
    """Iterate through the gaintables, logging resisual, combine into single GainTable

    :param gt_gen: Generator of GainTables
    :param do_plotting: Option to plot gain tables
    :param plot_dir: Directory to save plots

    :return: GainTable
    """
    gt_list = list()
    for gt in gt_gen:
        datetime = gt["datetime"][0].data
        log.info(
            f"rascil_rcal_gt_sink: Processing integration {datetime} residual: {numpy.max(gt.residual.data):.3g}"
        )
        gt_list.append(gt)

        if do_plotting == "True":
            gt_update_plot(gt_list, datetime, plot_dir)
            log.info("Updated gaintable plots.")

    full_gt = xarray.concat(gt_list, "time")
    return full_gt


def gt_update_plot(gt_list, datetime, plot_dir):

    """Plot gaintable (gain and residual values) over time

    :param gt_list: GainTable list to plot
    :param datetime: The current datetime
    :param plot_dir: Directory to save plots

    :return
    """

    with time_support(format="iso", scale="utc"):

        plt.cla()
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
        fig.subplots_adjust(hspace=0)

        gains = []
        residual = []
        time = []

        # We only look at the central antenna at the moment
        for gt in gt_list:

            time.append(gt.time.data[0] / 86400.0)
            current_gain = gt.gain.data[0]
            nchan = current_gain.shape[1]
            gains.append(current_gain[:, nchan // 2, 0, 0])
            residual.append(gt.residual.data[0, nchan // 2, 0, 0])

        # Plotting.
        gains = numpy.array(gains)
        amp = numpy.abs(gains)
        phase = numpy.angle(gains, deg=True)
        timeseries = Time(time, format="mjd", out_subfmt="str")
        for i in range(gains.shape[1]):
            ax1.plot(timeseries, amp[:, i] - 1.0, label=f"Antenna {i}")
            ax2.plot(timeseries, phase[:, i] - phase[:, 0], label=f"Antenna {i}")

        ax1.set_ylabel("Gain Amplitude - 1")
        ax2.set_ylabel("Gain Phase (Antenna - Antenna 0)")
        ax2.legend(loc="best")

        ax3.plot(timeseries, residual)
        ax3.set_ylabel("Residual")
        ax3.set_xlabel("Time(UTC)")
        ax3.set_yscale("log")
        plt.xticks(rotation=30)

        fig.suptitle(f"Updated GainTable at {datetime}")
        plt.savefig(plot_dir + f"plot_{datetime}.png")

    return


if __name__ == "__main__":
    parser = cli_parser()
    args = parser.parse_args()
    rcal_simulator(args)
