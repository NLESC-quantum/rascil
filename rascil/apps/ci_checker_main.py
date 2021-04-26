"""
Stand-alone application for finding sources with PyBDSF
"""

import argparse
import datetime
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import bdsf
import astropy.units as u
from astropy.coordinates import SkyCoord

from rascil.data_models import (
    PolarisationFrame,
    import_skycomponent_from_hdf5,
    export_skycomponent_to_hdf5,
)
from rascil.processing_components import create_low_test_skycomponents_from_gleam
from rascil.processing_components.skycomponent.operations import (
    create_skycomponent,
    find_skycomponent_matches,
    apply_beam_to_skycomponent,
)
from rascil.processing_components.image.operations import (
    import_image_from_fits,
    export_image_to_fits,
)
from rascil.processing_components.imaging.primary_beams import create_pb
from rascil.processing_components.skycomponent.plot_skycomponent import (
    plot_skycomponents_positions,
    plot_skycomponents_position_distance,
    plot_skycomponents_flux,
    plot_skycomponents_flux_ratio,
    plot_skycomponents_flux_histogram,
    plot_skycomponents_position_quiver,
    plot_gaussian_beam_position,
    plot_multifreq_spectral_index,
)

from rascil.apps.ci_checker.generate_results_index import create_index
from rascil.apps.ci_checker.ci_diagnostics import (
    ci_checker_diagnostics,
)


class FileFormatError(Exception):
    pass


log = logging.getLogger("rascil-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


def cli_parser():
    """Get a command line parser and populate it with arguments

    :param parser: argparse
    :return: CLI parser argparse
    """
    parser = argparse.ArgumentParser(
        description="RASCIL continuum imaging checker", fromfile_prefix_chars="@"
    )
    parser.add_argument(
        "--ingest_fitsname_restored",
        type=str,
        default=None,
        help="FITS file of the restored image to be read",
    )
    parser.add_argument(
        "--ingest_fitsname_residual",
        type=str,
        default=None,
        help="FITS file of the residual image to be read",
    )
    parser.add_argument(
        "--finder_beam_maj",
        type=float,
        default=1.0,
        help="Major axis of the restoring beam (degrees) (usually not needed, passed in restored image)",
    )
    parser.add_argument(
        "--finder_beam_min",
        type=float,
        default=1.0,
        help="Minor axis of the restoring beam (degrees) (usually not needed, passed in restored image)",
    )
    parser.add_argument(
        "--finder_beam_pos_angle",
        type=float,
        default=0.0,
        help="Positioning angle of the restoring beam (degrees) (usually not needed, passed in restored image)",
    )
    parser.add_argument(
        "--finder_th_isl",
        type=float,
        default=5.0,
        help="Threshold to determine the size of the islands",
    )
    parser.add_argument(
        "--finder_th_pix",
        type=float,
        default=10.0,
        help="Threshold to detect source (peak value)",
    )
    parser.add_argument(
        "--finder_multichan_option",
        type=str,
        default="single",
        help="For multi-channel images, what mode to perform BDSF on (single or average)",
    )
    parser.add_argument(
        "--apply_primary",
        type=str,
        default="False",
        help="Whether to apply primary beam",
    )
    parser.add_argument(
        "--telescope_model",
        type=str,
        default="MID",
        help="The telescope to generate primary beam correction",
    )
    parser.add_argument(
        "--check_source",
        type=str,
        default="False",
        help="Option to check with original input source catalogue",
    )
    parser.add_argument(
        "--plot_source",
        type=str,
        default="False",
        help="Option to plot position and flux errors for source catalogue",
    )
    parser.add_argument(
        "--input_source_format",
        type=str,
        default="external",
        help="The input format of the source catalogue",
    )
    parser.add_argument(
        "--input_source_filename",
        type=str,
        default=None,
        help="If use external source file, the file name of source file",
    )
    parser.add_argument(
        "--match_sep",
        type=float,
        default=1.0e-5,
        help="Maximum separation in radians for the source matching",
    )
    parser.add_argument(
        "--quiet_bdsf",
        type=str,
        default="False",
        help="If True, suppress bdsf.process_image() text output to screen. "
        "Output is still sent to the log file.",
    )
    parser.add_argument(
        "--source_file", type=str, default=None, help="Name of output source file"
    )
    parser.add_argument(
        "--rascil_source_file",
        type=str,
        default=None,
        help="Name of output RASCIL components hdf file",
    )
    parser.add_argument(
        "--logfile", type=str, default=None, help="Name of output log file"
    )
    parser.add_argument(
        "--savefits_rmsim",
        type=str,
        default="False",
        help="This parameter is a Boolean (default is False). If True, save background rms image as a FITS file.",
    )

    return parser


def analyze_image(args):
    """
    Main analysis routine.

    :param args: argparse with appropriate arguments

    :return:    A list of sources and their matches to original sources

    """

    if args.ingest_fitsname_restored is None:
        raise FileNotFoundError("Input restored FITS file name must be specified")

    if args.logfile is None:
        logfile = args.ingest_fitsname_restored.replace(".fits", ".log")
    else:
        logfile = args.logfile

    if args.source_file is None:
        source_file = args.ingest_fitsname_restored.replace(".fits", ".pybdsm.srl.csv")
    else:
        source_file = args.source_file

    def init_logging():
        logging.basicConfig(
            filename=logfile,
            filemode="a",
            format="%(asctime)s.%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%d/%m/%Y %I:%M:%S %p",
            level=logging.INFO,
        )

    init_logging()

    log.info("\nRASCIL Continuum Imaging Checker\n")

    starttime = datetime.datetime.now()
    log.info("Started : {}".format(starttime))
    log.info("Writing log to {}".format(logfile))

    input_image_restored = args.ingest_fitsname_restored

    im = import_image_from_fits(args.ingest_fitsname_restored)

    nchan = im["pixels"].shape[0]
    if nchan == 1:
        log.info("This is a single channel image.")
        freq = np.array([im.frequency.data[0]])

    elif nchan > 1:
        log.info("This is a multiple channel image.")
        freq = np.array(im.frequency.data)

    else:
        raise FileFormatError("This image is broken. Please check the file.")

    log.info("Frequencies of image:{} ".format(freq))

    # If read restoring beam from header
    try:
        beam_maj = im.attrs["clean_beam"]["bmaj"]
        beam_min = im.attrs["clean_beam"]["bmin"]
        beam_pos_angle = im.attrs["clean_beam"]["bpa"]

    except KeyError:

        beam_maj = args.finder_beam_maj
        beam_min = args.finder_beam_min
        beam_pos_angle = args.finder_beam_pos_angle

    beam_info = (beam_maj, beam_min, beam_pos_angle)

    th_isl = args.finder_th_isl
    th_pix = args.finder_th_pix

    log.info("Use restoring beam: {}".format(beam_info))
    log.info("Use threshold: {}, {}".format(th_isl, th_pix))

    input_image_residual = args.ingest_fitsname_residual
    quiet_bdsf = False if args.quiet_bdsf == "False" else True

    multichan_option = args.finder_multichan_option

    ci_checker(
        input_image_restored,
        input_image_residual,
        beam_info,
        source_file,
        th_isl,
        th_pix,
        nchan,
        multichan_option,
        quiet_bdsf=quiet_bdsf,
    )

    # check if there are sources found
    log.info("Output csv source file:{}".format(source_file))
    if os.path.exists(source_file) is False:
        log.error("Error: No source found. Please refine beam parameters.")
        return None, None

    if args.rascil_source_file is None:
        rascil_source_file = args.ingest_fitsname_restored.replace(
            ".fits", ".pybdsm.srl.hdf"
        )
    else:
        rascil_source_file = args.rascil_source_file

    log.info("Putting sources into skycomponents format.")
    out = create_source_to_skycomponent(source_file, rascil_source_file, freq)

    if args.check_source == "True":

        if args.input_source_format == "external":
            if (
                ".h5" in args.input_source_filename
                or ".hdf" in args.input_source_filename
            ):
                orig = import_skycomponent_from_hdf5(args.input_source_filename)

            elif ".txt" in args.input_source_filename:
                orig = read_skycomponent_from_txt(args.input_source_filename, freq)
            else:
                raise FileFormatError("Input file must be of format: hdf5 or txt.")
        else:  # Use internally provided GLEAM model
            orig = create_low_test_skycomponents_from_gleam(
                flux_limit=1.0,
                phasecentre=im.image_acc.phasecentre,
                frequency=freq,
                polarisation_frame=PolarisationFrame("stokesI"),
                radius=0.5,
            )

        # Compensate for primary beam correction -- NEEDS UPDATE
        if args.apply_primary == "True":
            telescope = args.telescope_model
            orig = add_primary_beam(input_image_restored, orig, telescope)

        results = check_source(orig, out, args.match_sep)

        if args.plot_source == "True":

            if len(results) == 0:
                log.info("No matches are found. Skipping plotting routines.")
            else:
                plot_file = args.ingest_fitsname_restored.replace(".fits", "")
                log.info("Plotting errors: {}".format(plot_file))
                plot_errors(orig, out, input_image_restored, args.match_sep, plot_file)

    else:
        results = None

    files_directory = os.path.dirname(args.ingest_fitsname_restored)
    if not files_directory:
        files_directory = os.getcwd()

    create_index(files_directory)

    log.info("Started  : {}".format(starttime))
    log.info("Finished : {}".format(datetime.datetime.now()))

    return out, results


def ci_checker(
    input_image_restored,
    input_image_residual,
    beam_info,
    source_file,
    th_isl,
    th_pix,
    nchan,
    multichan_option,
    quiet_bdsf=False,
):
    """
    PyBDSF-based source finder

    :param input_image
    :param input_image_residual
    :param beam_info : Size of restoring beam as (bmaj, bmin, pos_angle)
    :param source_file : Output file name of the source list
    :param th_isl : Island threshold
    :param th_pix: Peak threshold
    :param nchan: Number of channels
    :param multichan_option: Mode to perform BDSF on multi-channel images
    :param quiet_bdsf: if True, suppress text output of bdsf logs to screen.
                       Output is still sent to the log file
    : return None
    """

    # Process image.
    log.info("Analysing the restored image")

    refchan = nchan // 2
    log.info(
        "Found spectral cube with {} channel(s), using channel {} for source finding. The multi-channel BDSF mode is {}. ".format(
            nchan, refchan, multichan_option
        )
    )

    if nchan == 1:  # single frequency
        img_rest = bdsf.process_image(
            input_image_restored,
            beam=beam_info,
            thresh_isl=th_isl,
            thresh_pix=th_pix,
            quiet=quiet_bdsf,
        )
    else:

        img_rest = bdsf.process_image(
            input_image_restored,
            beam=beam_info,
            thresh_isl=th_isl,
            thresh_pix=th_pix,
            multichan_opts=True,
            collapse_mode=multichan_option,
            collapse_ch0=refchan,  # this only applies to single channel mode
            specind_maxchan=1,
            quiet=quiet_bdsf,
            spectralindex_do=True,
            specind_snr=5.0,
        )

    # Write the source catalog and the residual image.
    img_rest.write_catalog(
        outfile=source_file, format="csv", catalog_type="srl", clobber=True
    )
    img_rest.export_image(img_type="gaus_resid", clobber=True)

    log.info("Running diagnostics for the restored image")
    ci_checker_diagnostics(img_rest, input_image_restored, "restored")

    if input_image_residual is not None:
        log.info("Analysing the residual image")

        if nchan == 1:  # single frequency
            img_resid = bdsf.process_image(
                input_image_residual,
                beam=beam_info,
                thresh_isl=th_isl,
                thresh_pix=th_pix,
                quiet=quiet_bdsf,
            )
        else:
            img_resid = bdsf.process_image(
                input_image_residual,
                beam=beam_info,
                thresh_isl=th_isl,
                thresh_pix=th_pix,
                multichan_opts=True,
                collapse_mode=multichan_option,
                collapse_ch0=refchan,
                quiet=quiet_bdsf,
                spectralindex_do=True,
            )

        save_rms = input_image_residual.replace(".fits", "_residual_rms")

        if args.savefits_rmsim == "True":
            export_image_to_fits(img_resid.rms_arr, save_rms + ".fits")

        log.info("Running diagnostics for the residual image")
        ci_checker_diagnostics(img_resid, input_image_residual, "residual")

    return


def create_source_to_skycomponent(source_file, rascil_source_file, freq):
    """
    Put the sources into RASCIL-readable skycomponents

    :param source_file: Output file name of the source list
    :param rascil_source_file: Output file name of the RASCIL skycomponents hdf file
    :param freq: Single frequency or list of frequencies in float

    :return comp: List of skycomponents
    """

    data = pd.read_csv(source_file, sep=r"\s*,\s*", skiprows=5, engine="python")
    comp = []

    # TODO: Change this to multiple polarizaions
    nchan = len(freq)
    npol = 1
    centre = nchan // 2

    for i, row in data.iterrows():

        direc = SkyCoord(
            ra=row["RA"] * u.deg, dec=row["DEC"] * u.deg, frame="icrs", equinox="J2000"
        )
        f0 = row["Total_flux"]
        if f0 > 0:  # filter out ghost sources
            try:
                spec_indx = row["Spec_Indx"]
                fluxes = [f0 * (f / freq[centre]) ** spec_indx for f in freq]
                flux_array = np.reshape(np.array(fluxes), (nchan, npol))

            except KeyError:
                # No spectral index information
                flux_array = np.array([[f0]])

            comp.append(
                create_skycomponent(
                    direction=direc,
                    flux=flux_array,
                    frequency=freq,
                    polarisation_frame=PolarisationFrame("stokesI"),
                )
            )

    export_skycomponent_to_hdf5(comp, rascil_source_file)

    return comp


def add_primary_beam(input_image, comp, telescope):
    """
    Add optional primary beam correction for fluxes.

    :param input_image: Input image in FITS format
    :param comp: Source list in skycomponents format
    :param telescope: Telescope model to use, i.e. MID or LOW

    :return pbcomp: Corrected lists of skycomponents.

    """
    image = import_image_from_fits(input_image, fixpol=True)
    pb = create_pb(image, telescope=telescope, use_local=False)
    pbcomp = apply_beam_to_skycomponent(comp, pb)

    return pbcomp


def check_source(orig, comp, match_sep):
    """
    Check the difference between output sources and input.

    :param orig: Input source list in skycomponent format
    :param comp: Output source list in skycomponent format
    :param match_sep: The criteria for maximum separation

    :return matches: List of matched skycomponents

    """

    matches = find_skycomponent_matches(comp, orig, tol=match_sep)

    log.debug("Here is the complete list of matches.")

    for match in matches:
        m_comp = comp[match[0]]
        m_orig = orig[match[1]]
        log.debug(f"Original: {m_orig} Match {m_comp}")

    return matches


def read_skycomponent_from_txt(filename, freq):
    """
    Read source input from a txt file and make them into skycomponents

    :param filename: Name of input file
    :param freq: Frequency or list of frequencies in float
    :return comp: List of skycomponents
    """

    data = np.loadtxt(filename, delimiter=",", unpack=True)
    comp = []

    ra = data[0]
    dec = data[1]
    flux = data[2]
    ref_freq = data[6]
    spec_indx = data[7]

    nchan = len(freq)
    npol = 1
    for i, row in enumerate(ra):

        direc = SkyCoord(
            ra=ra[i] * u.deg, dec=dec[i] * u.deg, frame="icrs", equinox="J2000"
        )
        if nchan == 1:
            flux = flux[i] * (freq[0] / ref_freq[i]) ** spec_indx[i]
            flux_array = np.array([[flux]])
        else:
            fluxes = [flux[i] * (f / ref_freq[i]) ** spec_indx[i] for f in freq]
            flux_array = np.reshape(np.array(fluxes), (nchan, npol))

        comp.append(
            create_skycomponent(
                direction=direc,
                flux=flux_array,
                frequency=freq,
                polarisation_frame=PolarisationFrame("stokesI"),
            )
        )

    return comp


def plot_errors(orig, comp, input_image, match_sep, plot_file):
    """
    Plot the position and flux errors for source input and output

    :param orig: Input source list in skycomponent format
    :param comp: Output source list in skycomponent format
    :param input_image: Input image for Gaussian fits
    :param match_sep: The criteria for maximum separation
    :param plot_file: prefix of the plot files
    :return

    """
    log.info("Plotting skycomponents to check the accuracy of the source finder.")
    image = import_image_from_fits(input_image, fixpol=True)

    img_size = np.rad2deg(image.image_acc.wcs.wcs.cdelt[1])
    phasecentre = image.image_acc.phasecentre
    nchan = image["pixels"].shape[0]
    refchan = nchan // 2

    ra_comp, dec_comp = plot_skycomponents_positions(
        comp, orig, img_size=img_size, plot_file=plot_file, tol=match_sep
    )
    ra_error, dec_error = plot_skycomponents_position_distance(
        comp, orig, phasecentre, img_size, plot_file=plot_file, tol=match_sep
    )
    flux_in, flux_out = plot_skycomponents_flux(
        comp,
        orig,
        plot_file=plot_file,
        tol=match_sep,
        refchan=refchan,
    )
    dist, flux_ratio = plot_skycomponents_flux_ratio(
        comp,
        orig,
        phasecentre,
        plot_file=plot_file,
        tol=match_sep,
        refchan=refchan,
    )
    fluxes = plot_skycomponents_flux_histogram(
        comp,
        orig,
        plot_file=plot_file,
        tol=match_sep,
        refchan=refchan,
    )

    log.info("Plotting wide field plots.")
    ra_error, dec_error = plot_skycomponents_position_quiver(
        comp, orig, phasecentre, plot_file=plot_file, tol=match_sep
    )

    bmaj, bmin = plot_gaussian_beam_position(
        comp, orig, phasecentre, image, plot_file=plot_file, tol=match_sep
    )

    if nchan > 1:
        log.info("Plotting spectral index.")
        spec_in, spec_out = plot_multifreq_spectral_index(
            comp, orig, plot_file=plot_file, tol=match_sep
        )

    log.info("Plotting done.")

    return


if __name__ == "__main__":

    # Get command line inputs
    parser = cli_parser()
    args = parser.parse_args()
    analyze_image(args)
