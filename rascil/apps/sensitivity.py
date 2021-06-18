"""Simulation to calculate sensitivity

Run as e.g.

    python $RASCIL/rascil/apps/sensitivity.py --imaging_cellsize 2e-7 --imaging_npixel 1024 \
        --imaging_weighting uniform --rmax 1e5 --imaging_taper 6e-7
"""
import logging
import pprint
import sys
import argparse
import datetime

import matplotlib.pyplot as plt
import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models import PolarisationFrame

from rascil.processing_components import (
    create_image_from_visibility,
    create_blockvisibility,
    create_named_configuration,
    plot_uvcoverage,
    qa_image,
    export_image_to_fits,
    fit_psf,
)

from rascil.workflows import (
    invert_list_rsexecute_workflow,
    sum_invert_results_rsexecute,
    weight_list_rsexecute_workflow,
    taper_list_rsexecute_workflow,
)
from rascil.workflows.rsexecute.execution_support import rsexecute

pp = pprint.PrettyPrinter()


def init_logging():
    logging.basicConfig(
        filename="sensitivity.log",
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )


init_logging()
log = logging.getLogger("rascil-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


def cli_parser():
    """Get a command line parser and populate it with arguments

    First a CLI argument parser is created. Each function call adds more arguments to the parser.

    :return: CLI parser argparse
    """

    parser = argparse.ArgumentParser(
        description="Calculate relative sensitivity for MID observations",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--use_dask", type=str, default="True", help="Use dask processing?"
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
        help="Type of weighting: uniform or robust or natural",
    )
    parser.add_argument(
        "--imaging_robustness",
        type=float,
        default=0.0,
        help="Robustness for robust weighting",
    )
    parser.add_argument(
        "--imaging_taper",
        type=float,
        default=None,
        help="If set, use value for Gaussian taper, "
        "specified as radians in image plane",
    )

    parser.add_argument(
        "--ra", type=float, default=+15.0, help="Right ascension (degrees)"
    )
    parser.add_argument(
        "--declination", type=float, default=-45.0, help="Declination (degrees)"
    )
    parser.add_argument(
        "--rmax",
        type=float,
        default=2e5,
        help="Maximum distance of station from centre (m)",
    )
    parser.add_argument("--band", type=str, default="B2", help="Band")

    parser.add_argument(
        "--integration_time", type=float, default=740, help="Integration time (s)"
    )

    parser.add_argument(
        "--time_range",
        type=float,
        nargs=2,
        default=[-4.0, 4.0],
        help="Hour angle range in hours",
    )
    parser.add_argument("--nchan", type=int, default=12, help="Number of channels")

    parser.add_argument(
        "--channel_width", type=float, default=1.2e7, help="Channel bandwidth (Hz)"
    )

    parser.add_argument("--verbose", type=str, default="False", help="Verbose output?")

    return parser


def simulate_bvis(args):
    band = "B2"
    vis_polarisation_frame = PolarisationFrame("stokesI")

    import pprint

    pp = pprint.PrettyPrinter()

    def init_logging():
        logging.basicConfig(
            filename="sensitivity.log",
            filemode="a",
            format="%(asctime)s.%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%d/%m/%Y %I:%M:%S %p",
            level=logging.INFO,
        )

    init_logging()

    rsexecute.set_client(use_dask=args.use_dask)

    log.info("sensitivity: Starting MID sensitivity simulation\n")
    pp.pprint(vars(args))

    starttime = datetime.datetime.now()
    log.info("Started : {}".format(starttime))
    log.info("Writing log to {}".format("sensitivity.log"))

    rsexecute.run(init_logging)

    # Set up details of simulated observation
    if band == "B1":
        start_frequency = 0.765e9
    elif band == "B2":
        start_frequency = 1.36e9
    elif band == "Ku":
        start_frequency = 12.179e9
    else:
        raise ValueError("Unknown band %s" % band)

    frequency = numpy.linspace(
        start_frequency, start_frequency + args.nchan * args.channel_width, args.nchan
    )
    channel_bandwidth = numpy.array(args.nchan * [args.channel_width])

    phasecentre = SkyCoord(
        ra=args.ra * u.deg, dec=args.declination * u.deg, frame="icrs", equinox="J2000"
    )

    config = create_named_configuration("MIDR5", rmax=args.rmax)
    time_rad = numpy.array(args.time_range) * numpy.pi / 12.0
    times = numpy.arange(
        time_rad[0], time_rad[1], args.integration_time * numpy.pi / 43200.0
    )
    # Make a list of BlockVisibility's, one for each frequency.
    # This is actually a graph that we will compute later
    # Note that the weight of one sample is set to the time-bandwidth product
    bvis_list = [
        rsexecute.execute(create_blockvisibility)(
            config,
            times=times,
            phasecentre=phasecentre,
            polarisation_frame=vis_polarisation_frame,
            frequency=[frequency[channel]],
            channel_bandwidth=[channel_bandwidth[channel]],
            integration_time=args.integration_time,
            elevation_limit=numpy.deg2rad(15.0),
            weight=args.integration_time * args.channel_width,
        )
        for channel, freq in enumerate(frequency)
    ]
    bvis_list = rsexecute.persist(bvis_list)

    if args.verbose == "True":
        plt.clf()
        local_bvis_list = rsexecute.compute(bvis_list, sync=True)
        plot_uvcoverage(local_bvis_list)
        plt.show(block=False)

        log.info(
            f"Size of BlockVisibility for first channel: {local_bvis_list[0].nbytes * 2**-30:.6f}GB "
        )

    return bvis_list


def image_bvis(args, bvis_list):
    """Construct the PSF and calculate statistics

    :param args:
    :return: Results in dictionary
    """

    results = vars(args)

    # Now make the model images (actually a graph)

    model_list = [
        rsexecute.execute(create_image_from_visibility)(
            bvis,
            npixel=args.imaging_npixel,
            cellsize=args.imaging_cellsize,
        )
        for bvis in bvis_list
    ]

    # Apply weighting
    if args.imaging_weighting is not None:
        bvis_list = weight_list_rsexecute_workflow(
            bvis_list, model_list, weighting=args.imaging_weighting
        )

    # Apply Gaussian taper
    if args.imaging_taper is not None:
        bvis_list = taper_list_rsexecute_workflow(
            bvis_list,
            size_required=args.imaging_taper,
        )

    # Now we can make the PSF
    psf_list = invert_list_rsexecute_workflow(
        bvis_list, model_list, context="ng", dopsf=True, do_wstacking=False
    )
    result = sum_invert_results_rsexecute(psf_list, split=2)

    psf, sumwt = rsexecute.compute(result, sync=True)

    qa_psf = qa_image(psf)
    log.info(f"PSF QA = {qa_psf}")
    for key in qa_psf.data:
        results[f"psf_{key}"] = qa_psf.data[key]
    log.info(f"Total time-bandwidth product = {sumwt[0][0]:.3f}")

    results["tb_product"] = sumwt[0][0]

    if args.verbose == "True":
        export_image_to_fits(psf, "sensitivity_psf.fits")

    clean_beam = fit_psf(psf)

    log.info(f"Clean beam {clean_beam}")

    log.info("Finished : {}".format(datetime.datetime.now()))

    return results


if __name__ == "__main__":
    parser = cli_parser()
    args = parser.parse_args()
    bvis_list = simulate_bvis(args)
    results = image_bvis(args, bvis_list)
    log.info("Final results:")
    pp.pprint(results)
