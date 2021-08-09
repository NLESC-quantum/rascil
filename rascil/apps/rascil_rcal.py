""" RASCIL MS advice

"""

import argparse
import logging
import sys
import pprint
import numpy
import xarray

from rascil.data_models import BlockVisibility, GainTable, export_gaintable_to_hdf5
from rascil.processing_components import create_blockvisibility_from_ms, solve_gaintable

log = logging.getLogger("rascil-logger")
log.setLevel(logging.INFO)


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

    log.info(f"MS loaded into BlockVisibility:\n{bvis}\n")

    # This returns the full gaintable
    full_gt = gt_sink(bvis_solver_sink(bvis_source(bvis)))
    gtfile = args.ingest_msname.replace(".ms", "_gaintable.hdf")
    export_gaintable_to_hdf5(full_gt, gtfile)
    return gtfile


def bvis_source(bvis: BlockVisibility, dim="time") -> BlockVisibility:
    """

    :param bvis: BlockVisibility
    :return: BlockVisibility
    """
    log.info("rascil_rcal_bvis_source: BlockVisibility source starting")
    for time, bv in bvis.groupby(dim, squeeze=False):
        yield bv
    log.info("rascil_rcal_bvis_source: BlockVisibility source has finished")


def bvis_solver_sink(bvis):
    """Iterate through the block vis, solving for the gain

    :param bvis:
    :return: generator of gaintables
    """
    for bv in bvis:
        gt = solve_gaintable(bv, phase_only=True)
        yield gt


def gt_sink(gt_gen) -> GainTable:
    """Iterate through the gaintables

    :param gt_gen:
    :return: gaintable
    """
    gt_list = list()
    for gt in gt_gen:
        datetime = gt["datetime"][0].data
        log.info(
            f"rascil_rcal_gt_sink: Processing integration {datetime} residual: {numpy.max(gt.residual.data)}"
        )
        gt_list.append(gt)

    full_gt = xarray.concat(gt_list, "time")
    return full_gt


if __name__ == "__main__":
    parser = cli_parser()
    args = parser.parse_args()
    rcal_simulator(args)
