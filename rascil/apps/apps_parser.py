""" RASCIL CLI apps parser

Use these functions to construct a parser with standard arguments. For example::

    parser = argparse.ArgumentParser(description='RASCIL continuum imager')
    parser = apps_parser_app(parser)
    parser = apps_parser_ingest(parser)
    parser = apps_parser_imaging(parser)
    parser = apps_parser_calibration(parser)
    parser = apps_parser_cleaning(parser)
    parser = apps_parser_dask(parser)

The parser can then be called to extract the relevant command line argument::

    niter = args.cleaning_niter
    msname = args.ingest_msname

"""

import sys
import argparse
import json

def apps_parser_ingest(parser):
    """Add ingest-specific command line arguments to an existing CLI parser

    :param parser: argparse
    :return: argparse
    """
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
        "--ingest_vis_nchan",
        type=int,
        default=None,
        help="Number of channels in a single data descriptor in the MS",
    )
    parser.add_argument(
        "--ingest_chan_per_blockvis",
        type=int,
        default=1,
        help="Number of channels per blockvis (before any average)",
    )
    parser.add_argument(
        "--ingest_average_blockvis",
        type=str,
        default="False",
        help="Average all channels in blockvis?",
    )

    return parser


def apps_parser_imaging(parser):
    """Add imaging-specific command line arguments to an existing CLI parser

    :param parser: argparse
    :return: argparse
    """
    parser.add_argument(
        "--imaging_phasecentre",
        type=str,
        default=None,
        help="Phase centre (in SkyCoord string format)",
    )
    parser.add_argument(
        "--imaging_pol",
        type=str,
        default="stokesI",
        help="RASCIL polarisation frame for image",
    )
    parser.add_argument(
        "--imaging_nchan", type=int, default=1, help="Number of channels per image"
    )
    parser.add_argument(
        "--imaging_context",
        type=str,
        default="ng",
        help="Imaging context i.e. the gridder used 2d | ng",
    )
    parser.add_argument(
        "--imaging_ng_threads",
        type=int,
        default=4,
        help="Number of Nifty Gridder threads to use (4 is a good choice)",
    )
    parser.add_argument(
        "--imaging_w_stacking",
        type=str,
        default=True,
        help="Use the improved w stacking method in Nifty Gridder?",
    )
    parser.add_argument(
        "--imaging_flat_sky",
        type=str,
        default=False,
        help="If using a primary beam, normalise to flat sky?",
    )
    parser.add_argument(
        "--imaging_npixel",
        type=int,
        default=None,
        help="Number of pixels in ra, dec: Should be a composite of 2, 3, 5",
    )
    parser.add_argument(
        "--imaging_cellsize",
        type=float,
        default=None,
        help="Cellsize (radians). Default is to calculate.",
    )

    parser.add_argument(
        "--imaging_weighting",
        type=str,
        default="uniform",
        help="Type of weighting uniform or robust or natural)",
    )
    parser.add_argument(
        "--imaging_robustness",
        type=float,
        default=0.0,
        help="Robustness for robust weighting",
    )
    parser.add_argument(
        "--imaging_dopsf",
        type=str,
        default="False",
        help="Make the PSF instead of the dirty image?",
    )
    parser.add_argument(
        "--imaging_dft_kernel",
        type=str,
        default=None,
        help="DFT kernel: cpu_looped | cpu_numba | gpu_raw ",
    )

    return parser


def apps_parser_cleaning(parser):
    """Add cleaning-specific command line arguments to an existing CLI parser

    :param parser: argparse
    :return: argparse
    """

    parser.add_argument(
        "--clean_algorithm",
        type=str,
        default="mmclean",
        help="Type of deconvolution algorithm (hogbom or msclean or mmclean)",
    )
    parser.add_argument(
        "--clean_scales",
        type=int,
        nargs="*",
        default=[0],
        help="Scales for multiscale clean (pixels) e.g. [0, 6, 10]",
    )
    parser.add_argument(
        "--clean_nmoment",
        type=int,
        default=4,
        help="Number of frequency moments in mmclean (1 is a constant, 2 is linear, etc.)",
    )
    parser.add_argument(
        "--clean_nmajor",
        type=int,
        default=5,
        help="Number of major cycles in cip or ical",
    )
    parser.add_argument(
        "--clean_niter",
        type=int,
        default=1000,
        help="Number of minor cycles in CLEAN (i.e. clean iterations)",
    )
    parser.add_argument(
        "--clean_psf_support",
        type=int,
        default=256,
        help="Half-width of psf used in cleaning (pixels)",
    )
    parser.add_argument("--clean_gain", type=float, default=0.1, help="Clean loop gain")
    parser.add_argument(
        "--clean_threshold",
        type=float,
        default=1e-4,
        help="Clean stopping threshold (Jy/beam)",
    )
    parser.add_argument(
        "--clean_component_threshold",
        type=float,
        default=None,
        help="Sources > this level are converted to skycomponents",
    )
    parser.add_argument(
        "--clean_component_method",
        type=str,
        default="fit",
        help="Method to convert sources in image to skycomponents",
    )
    parser.add_argument(
        "--clean_fractional_threshold",
        type=float,
        default=0.3,
        help="Fractional stopping threshold for major cycle",
    )
    parser.add_argument(
        "--clean_facets",
        type=int,
        default=1,
        help="Number of overlapping facets in faceted clean (along each axis)",
    )
    parser.add_argument(
        "--clean_overlap",
        type=int,
        default=32,
        help="Overlap of facets in clean (pixels)",
    )
    parser.add_argument(
        "--clean_taper",
        type=str,
        default="tukey",
        help="Type of interpolation between facets in deconvolution (none or linear or tukey)",
    )
    parser.add_argument(
        "--clean_restore_facets",
        type=int,
        default=1,
        help="Number of overlapping facets in restore step (along each axis)",
    )
    parser.add_argument(
        "--clean_restore_overlap",
        type=int,
        default=32,
        help="Overlap of facets in restore step (pixels)",
    )
    parser.add_argument(
        "--clean_restore_taper",
        type=str,
        default="tukey",
        help="Type of interpolation between facets in restore step (none or linear or tukey)",
    )
    parser.add_argument(
        "--clean_restored_output",
        type=str,
        default="list",
        help="Type of restored image output: list or integrated",
    )

    return parser


def apps_parser_dask(parser):
    """Add Dask/rsexecute-specific command line arguments to an existing CLI parser

    :param parser: argparse
    :return: argparse
    """

    parser.add_argument(
        "--use_dask",
        type=str,
        default="True",
        help="Use Dask processing? False means that graphs are executed as they are constructed.",
    )
    parser.add_argument(
        "--dask_nthreads",
        type=int,
        default=None,
        help="Number of threads in each Dask worker (None means Dask will choose)",
    )
    parser.add_argument(
        "--dask_memory",
        type=str,
        default=None,
        help="Memory per Dask worker (GB), e.g. 5GB (None means Dask will choose)",
    )
    parser.add_argument(
        "--dask_nworkers",
        type=int,
        default=None,
        help="Number of workers (None means Dask will choose)",
    )
    parser.add_argument(
        "--dask_scheduler",
        type=str,
        default=None,
        help="Externally defined Dask scheduler e.g. 127.0.0.1:8786",
    )
    return parser


def apps_parser_slurm(parser):
    """Add SLURM-specific command line arguments to an existing CLI parser

    :param parser: argparse
    :return: argparse
    """

    parser.add_argument(
        "--use_slurm", type=str, default="False", help="Use the Dask SLURMCluster?"
    )
    parser.add_argument(
        "--slurm_project",
        type=str,
        default="SKA-SDP",
        help="SLURM project for accounting",
    )
    parser.add_argument(
        "--slurm_queue", type=str, default="compute", help="SLURM queue"
    )
    parser.add_argument(
        "--slurm_walltime", type=str, default="01:00:00", help="SLURM time limit"
    )
    return parser


def apps_parser_calibration(parser):
    """Add calibration-specific command line arguments to an existing CLI parser

    :param parser: argparse
    :return: argparse
    """
    parser.add_argument(
        "--calibration_T_first_selfcal",
        type=int,
        default=1,
        help="First selfcal for T (complex gain). T is common to both receptors",
    )
    parser.add_argument(
        "--calibration_T_phase_only",
        type=str,
        default="True",
        help="Phase only solution",
    )
    parser.add_argument(
        "--calibration_T_timeslice",
        type=float,
        default=None,
        help="Solution length (s) 0 means minimum",
    )

    parser.add_argument(
        "--calibration_G_first_selfcal",
        type=int,
        default=3,
        help="First selfcal for G (complex gain). G is different for the two receptors",
    )
    parser.add_argument(
        "--calibration_G_phase_only",
        type=str,
        default="False",
        help="Phase only solution?",
    )
    parser.add_argument(
        "--calibration_G_timeslice",
        type=float,
        default=None,
        help="Solution length (s) 0 means minimum",
    )

    parser.add_argument(
        "--calibration_B_first_selfcal",
        type=int,
        default=4,
        help="First selfcal for B (bandpass complex gain). B is complex gain per frequency.",
    )
    parser.add_argument(
        "--calibration_B_phase_only",
        type=str,
        default="False",
        help="Phase only solution",
    )
    parser.add_argument(
        "--calibration_B_timeslice",
        type=float,
        default=None,
        help="Solution length (s)",
    )

    parser.add_argument(
        "--calibration_global_solution",
        type=str,
        default="True",
        help="Solve across frequency",
    )
    parser.add_argument(
        "--calibration_context",
        type=str,
        default="T",
        help="Terms to solve (in order e.g. TGB)",
    )

    return parser


def apps_parser_app(parser):
    """Add app-specific command line arguments to an existing CLI parser

    :param parser: argparse
    :return: argparse
    """
    parser.add_argument(
        "--mode", type=str, default="cip", help="Processing  cip | ical | invert"
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default=None,
        help="Name of logfile (default is to construct one from msname)",
    )
    parser.add_argument(
        "--performance_file",
        type=str,
        default=None,
        help="Name of json file to contain performance information",
    )

    return parser
