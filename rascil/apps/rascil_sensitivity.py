"""Simulate MID observations to calculate point source and surface brightness sensitivity

Run as e.g.

    python $RASCIL/rascil/apps/rascil_sensitivity.py --imaging_cellsize 2e-7 --imaging_npixel 1024 \
        --imaging_weighting robust --imaging_robustness -2 -1 0 1 2 --rmax 1e5 --imaging_taper 0.0 6e-7 1.2e-6
"""
import logging
import pprint
import sys
import argparse
import datetime
import pandas as pd
import json

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
        "--efficiency", type=float, default=1.0, help="Correlator efficiency"
    )
    parser.add_argument(
        "--diameter", type=float, default=15.0, help="MID antenna diameter (m)"
    )
    parser.add_argument(
        "--declination", type=float, default=-45.0, help="Declination (degrees)"
    )
    parser.add_argument(
        "--configuration",
        type=str,
        default="MIDR5",
        help="Name of configuration or path: MID(=MIDR5), MIDR5, MEERKAT+",
    )
    parser.add_argument(
        "--subarray",
        type=str,
        default="",
        help="Name of file containing json description of subarray to be used",
    )
    parser.add_argument(
        "--rmax",
        type=float,
        default=2e5,
        help="Maximum distance of station from centre (m)",
    )
    parser.add_argument(
        "--frequency", type=float, default=1.36e9, help="Centre frequency (Hz)"
    )

    parser.add_argument(
        "--integration_time", type=float, default=600, help="Integration time (s)"
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
        "--channel_width", type=float, default=1e8, help="Channel bandwidth (Hz)"
    )

    parser.add_argument("--verbose", type=str, default="False", help="Verbose output?")

    parser.add_argument(
        "--results",
        type=str,
        default="rascil_sensitivity",
        help="Root name for output files",
    )

    return parser


def calculate_sensitivity(args):
    """Perform the full calculation of sensitivity, write results to csv file

    :param args:
    :return:
    """

    vis_polarisation_frame = PolarisationFrame("stokesI")

    pp = pprint.PrettyPrinter()

    rsexecute.set_client(use_dask=args.use_dask == "True")

    log.info("rascil_sensitivity: Starting MID sensitivity simulation\n")
    pp.pprint(vars(args))

    starttime = datetime.datetime.now()
    log.info("Started : {}".format(starttime))
    log.info("Writing log to {}".format("sensitivity.log"))

    rsexecute.run(init_logging)

    start_frequency = args.frequency

    frequency = numpy.linspace(
        start_frequency, start_frequency + args.nchan * args.channel_width, args.nchan
    )
    channel_bandwidth = numpy.array(args.nchan * [args.channel_width])

    phasecentre = SkyCoord(
        ra=args.ra * u.deg, dec=args.declination * u.deg, frame="icrs", equinox="J2000"
    )

    # Set up configuration and observing times
    config = create_named_configuration(args.configuration)
    print("subarray in", args.subarray)
    if args.subarray != "":
        f = open(args.subarray)
        subarray_dict = json.load(f)
        f.close()
        subarray_ids = subarray_dict["ids"]
        print(subarray_ids)
        config = config.sel({"id": subarray_ids})
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

    results = image_bvis(args, bvis_list)

    return results


def image_bvis(args, bvis_list):
    """Construct the PSF and calculate statistics

    :param args: CLI arguments
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

    log.info("Final results:")
    results_file = save_results(args, results)

    log.info("Finished : {}".format(datetime.datetime.now()))
    log.info(f"Results are in {results_file}")

    return results_file


def robustness_taper_scenario(
    args,
    weighting,
    robustness,
    taper,
    bvis_list,
    model_list,
):
    """Grid the data, form the PSF, and calculate the noise characteristics

    :param args: CLI args
    :param bvis_list: List of blockvisibility's
    :param model_list: List of models, one for each blockvisibility
    :return: Results in a dict
    """
    results = dict()

    # Apply weighting
    if weighting == "natural":
        # Natural
        bvis_list = weight_list_rsexecute_workflow(
            bvis_list,
            model_list,
            weighting="natural",
        )
        results["weighting"] = "natural"
        results["robustness"] = 0.0
    else:
        # Robust: first we reset the weights to natural
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

    # Now we can make the PSF.
    psf_list = invert_list_rsexecute_workflow(
        bvis_list,
        model_list,
        context="ng",
        dopsf=True,
    )
    # Sum all PSFs into one PSF
    result = sum_invert_results_rsexecute(psf_list)
    psf, sumwt = rsexecute.compute(result, sync=True)
    log.info(f"\nWeighting {weighting} robustness {robustness}, taper {taper} (radian)")
    if args.verbose == "True":
        export_image_to_fits(
            psf,
            f"{args.results}_sensitivity_{weighting}_robustness{robustness}_taper{taper}.fits",
        )

    clean_beam = fit_psf(psf)
    log.info(f"\tClean beam {clean_beam} (degrees)")
    for key in clean_beam:
        results[f"cleanbeam_{key}"] = clean_beam[key]

    results["sum_weights"] = sumwt
    qa_psf = qa_image(psf)
    for key in qa_psf.data:
        results[f"psf_{key}"] = qa_psf.data[key]

    # The effective time-bandwidth product (i.. accounting for weighting and taper)
    tb = sumwt[0][0] + sumwt[-1][0]
    log.info(f"\tTime-Bandwidth product (tb) = {tb:.4g} (Hz.s)")

    # Point source sensitivity
    # Equation 6.50 of Thompson, Moran, and Swenson
    d = args.diameter
    area = numpy.pi * (d / 2.0) ** 2
    efficiency = args.efficiency
    pss = (
        numpy.sqrt(2.0) * 1e26 * k_B * args.tsys / (area * efficiency * numpy.sqrt(tb))
    )
    results["pss"] = pss
    log.info(f"\tPoint source sensitivity (pss) = {pss:.4g} (Jy/(clean beam))")

    # Calculate solid angle of clean beam
    solid_angle = (
        1.1331 * numpy.deg2rad(clean_beam["bmaj"]) * numpy.deg2rad(clean_beam["bmaj"])
    )
    results["sa"] = solid_angle
    log.info(
        f"\tSolid angle of clean beam (sa) = {solid_angle:.4g} (steradian/(clean beam))"
    )

    # Calculate surface brightness visibility
    sbs = pss / solid_angle
    results["sbs"] = sbs
    log.info(f"\tSurface brightness sensitivity (sbs) = {sbs:.4g} (Jy/steradian)")
    results["tb"] = tb

    return results


def save_results(args, results):
    """Save the results to a CSV file

    :param args:
    :param results:
    :return:
    """

    df = pd.DataFrame(results)
    log.info(df)
    results_file = f"{args.results}_sensitivity.csv"
    df.to_csv(results_file, index=False)
    return results_file


if __name__ == "__main__":
    parser = cli_parser()
    args = parser.parse_args()
    results_file = calculate_sensitivity(args)
