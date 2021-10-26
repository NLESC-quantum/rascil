"""Simulation to calculate sensitivity

Run as e.g.

    python $RASCIL/rascil/apps/rascil_sensitivity.py --imaging_cellsize 2e-7 --imaging_npixel 1024 \
        --imaging_weighting uniform --rmax 1e5 --imaging_taper 6e-7
"""
import logging
import pprint
import sys
import argparse
import datetime
import pandas as pd

import matplotlib.pyplot as plt
import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.constants import Boltzmann as k_B

from rascil.data_models import PolarisationFrame

from rascil.processing_components import (
    create_image_from_visibility,
    create_blockvisibility,
    create_named_configuration,
    plot_uvcoverage,
    qa_image,
    export_image_to_fits,
    fit_psf,
    advise_wide_field,
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
        default=1024,
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
        default="robust",
        help="Type of weighting: uniform or robust or natural",
    )
    parser.add_argument(
        "--imaging_robustness",
        type=float,
        nargs="*",
        default=[-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        help="Robustness for robust weighting",
    )
    parser.add_argument(
        "--imaging_taper",
        nargs="*",
        type=float,
        default=None,
        help="If set, use value for Gaussian taper, "
        "specified as radians in image plane",
    )

    parser.add_argument(
        "--ra", type=float, default=+15.0, help="Right ascension (degrees)"
    )
    parser.add_argument(
        "--tsys", type=float, default=20.0, help="System temperature (K)"
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
    parser.add_argument("--nchan", type=int, default=1, help="Number of channels")

    parser.add_argument(
        "--channel_width", type=float, default=1e6, help="Channel bandwidth (Hz)"
    )

    parser.add_argument("--verbose", type=str, default="False", help="Verbose output?")

    parser.add_argument(
        "--results",
        type=str,
        default="rascil_sensitivity",
        help="Root name for output files",
    )

    return parser


def simulate_bvis(args):
    band = "B2"
    vis_polarisation_frame = PolarisationFrame("stokesI")

    pp = pprint.PrettyPrinter()

    rsexecute.set_client(use_dask=args.use_dask == "True")

    log.info("rascil_sensitivity: Starting MID sensitivity simulation\n")
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

    # Set up configuration and observing times
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

    # If the cellsize has not been specified, calculate it.
    if args.imaging_cellsize is None:
        advice = [
            rsexecute.execute(advise_wide_field)(bvis, verbose=False)
            for bvis in bvis_list
        ]
        advice = rsexecute.compute(advice, sync=True)
        cellsize = min(advice[0]["cellsize"], advice[-1]["cellsize"])
    else:
        cellsize = args.imaging_cellsize

    # Now make the model images (actually a graph)
    model_list = [
        rsexecute.execute(create_image_from_visibility)(
            bvis,
            npixel=args.imaging_npixel,
            cellsize=cellsize,
        )
        for bvis in bvis_list
    ]

    # Handle input arguments
    results = list()
    robustnesses = args.imaging_robustness
    if robustnesses is None:
        robustnesses = [None]
    tapers = args.imaging_taper
    if tapers is None:
        tapers = [0.0]

    for taper in tapers:
        result = robustness_taper_scenario(
            args, "uniform", 0.0, taper, bvis_list, model_list
        )
        results.append(result)
        for robustness in robustnesses:
            result = robustness_taper_scenario(
                args, "robust", robustness, taper, bvis_list, model_list
            )
            results.append(result)
        result = robustness_taper_scenario(
            args, "natural", 0.0, taper, bvis_list, model_list
        )
        results.append(result)

    log.info("Finished : {}".format(datetime.datetime.now()))

    return results


def robustness_taper_scenario(
    args,
    weighting,
    robustness,
    taper,
    bvis_list,
    model_list,
):
    """Grid the data, form the PSF, and calculate the noise characteristics

    :param args:
    :param bvis_list: List of blockvisibility's
    :param model_list: List of models, one for each blockvisibility
    :return: Results in a dict
    """
    results = dict()

    # Apply weighting, first we reset the weights to natural
    if weighting == "natural":
        bvis_list = weight_list_rsexecute_workflow(
            bvis_list,
            model_list,
            weighting="natural",
        )
        results["weighting"] = "natural"
        results["robustness"] = 0.0
    else:
        bvis_list = weight_list_rsexecute_workflow(
            bvis_list,
            model_list,
            weighting="natural",
        )
        bvis_list = weight_list_rsexecute_workflow(
            bvis_list,
            model_list,
            weighting=weighting,
            robustness=robustness,
        )
        results["weighting"] = weighting
        results["robustness"] = robustness

    # Apply Gaussian taper

    if taper > 0.0:
        bvis_list = taper_list_rsexecute_workflow(
            bvis_list,
            size_required=taper,
        )
    results["taper"] = taper

    # Now we can make the PSF
    psf_list = invert_list_rsexecute_workflow(
        bvis_list, model_list, context="ng", dopsf=True, do_wstacking=False
    )
    result = sum_invert_results_rsexecute(psf_list)
    psf, sumwt = rsexecute.compute(result, sync=True)
    log.info(f"\nWeighting {weighting} robustness {robustness}, taper {taper} (radian)")
    if args.verbose == "True":
        export_image_to_fits(
            psf,
            f"{args.results}_weighting{weighting}__robustness{robustness}_taper{taper}.fits",
        )

    clean_beam = fit_psf(psf)
    log.info(f"\tClean beam {clean_beam} (degrees)")
    for key in clean_beam:
        results[f"\tcleanbeam_{key}"] = clean_beam[key]

    results["sum_weights"] = sumwt
    qa_psf = qa_image(psf)
    for key in qa_psf.data:
        results[f"psf_{key}"] = qa_psf.data[key]

    # Time-bandwidth product
    tb = sumwt[0][0] + sumwt[-1][0]
    log.info(f"\tTime-Bandwidth product (tb) = {tb:.4g} (Hz.s)")

    # Point source sensitivty
    # Equation 6.50 of Thompson, Moran, and Swenson
    d = rsexecute.execute(lambda bv: bv.configuration.diameter[0].data)(bvis_list[0])
    d = rsexecute.compute(d, sync=True)

    area = numpy.pi * (d / 2.0) ** 2
    efficiency = 1.0

    pss = (
        numpy.sqrt(2.0) * 1e26 * k_B * args.tsys / (area * efficiency * numpy.sqrt(tb))
    )
    results["pss"] = pss
    log.info(f"\tPoint source sensitivity (pss) = {pss:.4g} (Jy/(clean beam))")

    solid_angle = (
        1.1331 * numpy.deg2rad(clean_beam["bmaj"]) * numpy.deg2rad(clean_beam["bmaj"])
    )
    sbs = pss / solid_angle
    results["sa"] = solid_angle
    log.info(
        f"\tSolid angle of clean beam (sa) = {solid_angle:.4g} (steradian/(clean beam))"
    )

    results["sbs"] = sbs
    log.info(f"\tSurface brightness sensitivity (sbs) = {sbs:.4g} (Jy/steradian)")
    results["tb"] = tb

    return results


def save_results(args, results):

    df = pd.DataFrame(results)
    log.info(df)
    df.to_csv(f"{args.results}.csv")
    return df


if __name__ == "__main__":
    parser = cli_parser()
    args = parser.parse_args()
    bvis_list = simulate_bvis(args)
    results = image_bvis(args, bvis_list)
    log.info("Final results:")
    save_results(args, results)
