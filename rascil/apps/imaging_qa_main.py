"""
Stand-alone application for finding sources with PyBDSF
"""

import argparse
import datetime
import logging
import os
import sys
import glob

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import bdsf
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel

from rascil.data_models import (
    PolarisationFrame,
    import_skycomponent_from_hdf5,
    export_skycomponent_to_hdf5,
)
from rascil.processing_components.skycomponent.operations import (
    create_skycomponent,
    find_skycomponent_matches,
    apply_beam_to_skycomponent,
)
from rascil.processing_components.image.operations import (
    import_image_from_fits,
    export_image_to_fits,
    create_image_from_array,
    add_image,
    qa_image,
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

from rascil.apps.imaging_qa.generate_results_index import create_index
from rascil.apps.imaging_qa.imaging_qa_diagnostics import (
    imaging_qa_diagnostics,
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
        "--ingest_fitsname_sensitivity",
        type=str,
        default=None,
        help="FITS file of the sensitivity image to be read",
    )
    parser.add_argument(
        "--ingest_fitsname_moment",
        type=str,
        default=None,
        help="FITS file of the frequency moment images to be read "
        "(Note: Use the prefix of the fits files, e.g. if the restored image is "
        "test_image_restored.fits here should input test_image)",
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
        "--finder_thresh_isl",
        type=float,
        default=5.0,
        help="Threshold to determine the size of the islands used in BDSF (Blob Detector and Source Finder)",
    )
    parser.add_argument(
        "--finder_thresh_pix",
        type=float,
        default=10.0,
        help="Threshold to detect source (peak value) used in BDSF",
    )
    parser.add_argument(
        "--finder_multichan_option",
        type=str,
        default="single",
        help="For multi-channel images, what mode to perform source detection on (single or average)",
    )
    parser.add_argument(
        "--perform_diagnostics",
        type=str,
        default="False",
        help="Whether to perform diagnostics of the images (restored and residual)",
    )
    parser.add_argument(
        "--apply_primary",
        type=str,
        default="False",
        help="Whether to divide by primary beam after BDSF to correct source flux",
    )
    parser.add_argument(
        "--use_frequency_moment",
        type=str,
        default="False",
        help="Whether to use frequency moment images after BDSF to correct spectral index",
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
        "--flux_limit",
        type=float,
        default=1.0e-3,
        help="Minimum flux where comparison plots are generated",
    )
    parser.add_argument(
        "--trim_image",
        type=str,
        default="False",
        help="For spectral index calculation, do we trim the image to avoid the edge effects?",
    )
    parser.add_argument(
        "--trim_box",
        type=float,
        default=3.0e-2,
        help="If trim_image is true, proportion of the box that is trimmed (default is 3%%)",
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
    parser.add_argument(
        "--restart",
        type=str,
        default="False",
        help="If true, surpass BDSF when the output already exists. "
        "The checker will start from reading the BDSF csv file",
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

    image_pol = im._polarisation_frame
    log.info("Frequencies of image:{} ".format(freq))
    log.info("Image polarisation frame is: {}".format(image_pol))

    input_image_residual = args.ingest_fitsname_residual
    input_image_sensitivity = args.ingest_fitsname_sensitivity
    quiet_bdsf = False if args.quiet_bdsf == "False" else True
    perform_diagnostics = False if args.perform_diagnostics == "False" else True
    restart = args.restart
    saverms = False if args.savefits_rmsim == "False" else True
    multichan_option = args.finder_multichan_option

    # If read restoring beam from header
    try:
        beam_maj = im.attrs["clean_beam"]["bmaj"]
        beam_min = im.attrs["clean_beam"]["bmin"]
        beam_pos_angle = im.attrs["clean_beam"]["bpa"]

    except TypeError:

        beam_maj = args.finder_beam_maj
        beam_min = args.finder_beam_min
        beam_pos_angle = args.finder_beam_pos_angle

    beam_info = (beam_maj, beam_min, beam_pos_angle)

    thresh_isl = args.finder_thresh_isl
    thresh_pix = args.finder_thresh_pix

    log.info("Use restoring beam: {}".format(beam_info))
    log.info("Use threshold: {}, {}".format(thresh_isl, thresh_pix))

    if restart == "False":
        imaging_qa_bdsf(
            input_image_restored,
            input_image_residual,
            beam_info,
            source_file,
            thresh_isl,
            thresh_pix,
            nchan,
            multichan_option,
            perform_diagnostics=perform_diagnostics,
            quiet_bdsf=quiet_bdsf,
            saverms=saverms,
        )
    else:
        log.info("Restart option is on. Will directly read from the source file.")

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

    if args.ingest_fitsname_moment is not None:

        # Used for RASCIL images
        input_image_moment = args.ingest_fitsname_moment + "_taylor[1-9].fits"

    else:

        # Used for YANDA image format
        input_image_moment = args.ingest_fitsname_restored.replace(".0.", ".[1-9].")

    moment_images = glob.glob(input_image_moment)
    log.info("Number of frequency moments image found: {}".format(len(moment_images)))

    log.info("Putting sources into skycomponents format.")
    out = create_source_to_skycomponent(source_file, rascil_source_file, freq)

    # Correct and put into new csv file
    # Calculate spectral index from frequency moment images
    if args.use_frequency_moment == "True" and len(moment_images) != 0:

        log.info("Calculate spectral index from frequency moment images.")

        if args.trim_image == "True":
            box_size = 1.0 - args.trim_box
        else:
            box_size = 1.0
        out = calculate_spec_index_from_moment(
            out, moment_images, box_size=box_size, flux_limit=args.flux_limit
        )

    # Compensate for primary beam correction
    # Note this should be applied to source out when it is division, source in when multiplication
    if args.apply_primary == "True":
        log.info("Correcting fluxes for primary beam.")
        telescope = args.telescope_model
        out = correct_primary_beam(
            input_image_restored, input_image_sensitivity, out, telescope=telescope
        )

    if args.check_source == "True":

        # Placeholder: Currently only supports comparsion for StokesI images
        if image_pol != "stokesI":
            log.error("Currently only supports StokesI images.")
            results = None

        if ".h5" in args.input_source_filename or ".hdf" in args.input_source_filename:
            orig = import_skycomponent_from_hdf5(args.input_source_filename)

        elif ".txt" in args.input_source_filename:
            orig = read_skycomponent_from_txt(args.input_source_filename, freq)
        else:
            raise FileFormatError("Input file must be of format: hdf5 or txt.")

        results = check_source(orig, out, args.match_sep)

        if args.plot_source == "True":

            if len(results) == 0:
                log.info("No matches are found. Skipping plotting routines.")
            else:
                plot_file = args.ingest_fitsname_restored.replace(".fits", "")
                if args.use_frequency_moment == "True" and len(moment_images) != 0:
                    csv_name = moment_images[0].replace(".fits", "_corrected.csv")
                else:
                    csv_name = None
                log.info("Plotting errors: {}".format(plot_file))
                plot_errors(
                    orig,
                    out,
                    input_image_restored,
                    args.match_sep,
                    sources_in_file=args.input_source_filename,
                    sources_out_file=csv_name,
                    flux_limit=args.flux_limit,
                    plot_file=plot_file,
                )

    else:
        results = None

    files_directory = os.path.dirname(args.ingest_fitsname_restored)
    if not files_directory:
        files_directory = os.getcwd()

    create_index(files_directory)

    log.info("Started  : {}".format(starttime))
    log.info("Finished : {}".format(datetime.datetime.now()))

    return out, results


def imaging_qa_bdsf(
    input_image_restored,
    input_image_residual,
    beam_info,
    source_file,
    thresh_isl,
    thresh_pix,
    nchan,
    multichan_option,
    perform_diagnostics=False,
    quiet_bdsf=False,
    saverms=False,
):
    """
    PyBDSF-based source finder

    :param input_image
    :param input_image_residual
    :param beam_info : Size of restoring beam as (bmaj, bmin, pos_angle)
    :param source_file : Output file name of the source list
    :param thresh_isl : Island threshold
    :param thresh_pix: Peak threshold
    :param nchan: Number of channels
    :param multichan_option: Mode to perform BDSF on multi-channel images
    :param perform_diagnostics: if True, perform diagnostics function on the images
    :param quiet_bdsf: if True, suppress text output of bdsf logs to screen.
                       Output is still sent to the log file
    :param saverms: if True, save background rms image as a FITS file
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

    before_time = datetime.datetime.now()
    log.info("Going into BDSF at time : {}".format(before_time))

    if nchan == 1:  # single frequency
        img_rest = bdsf.process_image(
            input_image_restored,
            beam=beam_info,
            thresh_isl=thresh_isl,
            thresh_pix=thresh_pix,
            quiet=quiet_bdsf,
        )
    else:

        img_rest = bdsf.process_image(
            input_image_restored,
            beam=beam_info,
            thresh_isl=thresh_isl,
            thresh_pix=thresh_pix,
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

    after_time = datetime.datetime.now()
    log.info("Exiting BDSF at time : {}".format(after_time))

    if perform_diagnostics:
        log.info("Running diagnostics for the restored image")
        imaging_qa_diagnostics(img_rest, input_image_restored, "restored")

    if input_image_residual is not None:
        log.info("Analysing the residual image")

        if nchan == 1:  # single frequency
            img_resid = bdsf.process_image(
                input_image_residual,
                beam=beam_info,
                thresh_isl=thresh_isl,
                thresh_pix=thresh_pix,
                quiet=quiet_bdsf,
            )
        else:
            img_resid = bdsf.process_image(
                input_image_residual,
                beam=beam_info,
                thresh_isl=thresh_isl,
                thresh_pix=thresh_pix,
                multichan_opts=True,
                collapse_mode=multichan_option,
                collapse_ch0=refchan,
                quiet=quiet_bdsf,
                spectralindex_do=True,
            )

        if saverms:

            save_rms_file = input_image_residual.replace(".fits", "_residual_rms")
            residual_im = import_image_from_fits(input_image_residual)

            resid_im_array = np.reshape(
                img_resid.rms_arr,
                (1, 1, img_resid.rms_arr.shape[1], img_resid.rms_arr.shape[0]),
            )
            rms_image = create_image_from_array(
                resid_im_array,
                residual_im.image_acc.wcs,
                residual_im.image_acc.polarisation_frame,
            )
            export_image_to_fits(rms_image, save_rms_file + ".fits")

        if perform_diagnostics:

            log.info("Running diagnostics for the residual image")
            imaging_qa_diagnostics(img_resid, input_image_residual, "residual")

    return


def create_source_to_skycomponent(source_file, rascil_source_file, freq):
    """
    Put the sources into RASCIL-readable skycomponents

    :param source_file: Output file name of the source list
    :param rascil_source_file: Output file name of the RASCIL skycomponents hdf file
    :param freq: List of frequencies in float
                 (if single frequency, pass it in a length one array)
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
        f0 = row["Peak_flux"]
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


def calculate_spec_index_from_moment(
    comp_list, moment_images, box_size=1.0, flux_limit=1.0e-3
):
    """
    Calculate spectral index using frequency moment images.

    :param comp_list: Source list in skycomponents format
    :param moment_images: Frequency moment images in FITS format
    :param box_size: Size of the box to perform the calculation
    :param flux_limit: The lower limit of flux (for the sources) over which spectral index is calculated

    :return newcomp: New lists of skycomponents with updated spectral index.

    """

    log.info("Use box: {} and flux_limit: {}".format(box_size, flux_limit))

    if len(moment_images) == 0:
        log.info("No moment images found, no csv file written.")
        return comp_list

    else:

        moment_data = import_image_from_fits(moment_images[0])

        # This applies to multiple Taylor images (not tested now)
        #        if len(moment_images) > 1:
        #            for moment_image in moment_images:
        #                moment_data_now = import_image_from_fits(moment_image)
        #                moment_data = add_image(moment_data, moment_data_now)

        image_frequency = moment_data.frequency.data
        nchan = len(comp_list[0].frequency.data)
        if nchan != len(image_frequency):
            log.info("The frequencies don't match.")
            return comp_list

        else:
            ras = [comp.direction.ra.degree for comp in comp_list]
            decs = [comp.direction.dec.degree for comp in comp_list]
            skycoords = SkyCoord(ras * u.deg, decs * u.deg, frame="icrs")
            pixlocs = skycoord_to_pixel(
                skycoords, moment_data.image_acc.wcs, origin=0, mode="wcs"
            )

            # TODO: Needs update for multiple polarizations
            npol = 1
            npixel = moment_data["pixels"].data.shape[2]
            freqs = [comp.frequency.data[nchan // 2] for comp in comp_list]
            central_fluxes = [comp.flux[nchan // 2][0] for comp in comp_list]

            spec_indx = np.zeros(len(comp_list))

            # Calculate spectral index when
            # 1) The flux is above flux_limit
            # 2) The frequency matches
            # 3) The pixel is within box_size
            for icomp, comp in enumerate(comp_list):

                if central_fluxes[icomp] > flux_limit:

                    pixloc = (round(pixlocs[0][icomp]), round(pixlocs[1][icomp]))

                    if (
                        np.max(np.abs(comp.frequency.data - image_frequency)) < 1e-7
                        and pixloc[0] < box_size * npixel
                        and pixloc[1] < box_size * npixel
                    ):
                        flux = moment_data["pixels"].data[
                            nchan // 2, 0, pixloc[1], pixloc[0]
                        ]

                        log.debug(
                            "Taylor flux:{} for skycomponent {}, {}, compared to original flux {}".format(
                                flux, ras[icomp], decs[icomp], central_fluxes[icomp]
                            )
                        )
                        spec_indx[icomp] = flux / central_fluxes[icomp]

                log.debug("Spectral index calculated is {}".format(spec_indx[icomp]))
                fluxes = [
                    comp.flux[i][0]
                    * (f / comp.frequency.data[nchan // 2]) ** spec_indx[icomp]
                    for i, f in enumerate(comp.frequency.data)
                ]
                flux_array = np.reshape(np.array(fluxes), (nchan, npol))
                comp.flux = flux_array

                # Write to new csv file
            ds = pd.DataFrame(
                {
                    "RA (deg)": ras,
                    "Dec (deg)": decs,
                    "Central freq (Hz)": freqs,
                    "Central flux (Jy)": central_fluxes,
                    "Spectral index": spec_indx,
                }
            )
            csv_name = moment_images[0].replace(".fits", "_corrected.csv")
            log.info("Writing source data to {}".format(csv_name))

            ds.to_csv(csv_name, index=True)

            return comp_list


def correct_primary_beam(input_image, sensitivity_image, comp, telescope="MID"):
    """
    Add optional primary beam correction for fluxes.

    :param input_image: Input image in FITS format
    :param sensitivity_image: Input sensitivity image in FITS format
    :param comp: Source list in skycomponents format
    :param telescope: Telescope model to use, i.e. MID or LOW

    :return pbcomp: Corrected lists of skycomponents.

    """

    if sensitivity_image is not None:
        log.info("Use beam from {}".format(sensitivity_image))
        beam = import_image_from_fits(sensitivity_image)

    elif input_image is not None:
        # Use internally provided telescope primary beam
        image = import_image_from_fits(input_image)
        beam = create_pb(
            image,
            telescope=telescope,
            pointingcentre=image.image_acc.phasecentre,
            use_local=False,
        )
    else:
        log.warning(
            "Please provide either the sensitivity image or the restored image."
        )
        return comp

    nchan, npol, ny, nx = beam["pixels"].data.shape
    log.info("The primary beam has {} channels, {} polarizaion.".format(nchan, npol))

    pbcomp = apply_beam_to_skycomponent(comp, beam, inverse=True)

    log.debug("Flux comparison after primary beam correction:")
    for i, c in enumerate(comp):
        log.debug(
            "Original {}, corrected {} \n".format(
                c.flux[nchan // 2][0], pbcomp[i].flux[nchan // 2][0]
            )
        )

    return pbcomp


def check_source(orig, comp, match_sep):
    """
    Check the difference between output sources and input.

    :param orig: Input source list in skycomponent format
    :param comp: Output source list in skycomponent format
    :param match_sep: The criteria for maximum separation

    :return matches: List of matched skycomponents

    """

    log.info("{} sources in and {} sources out".format(len(orig), len(comp)))

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
    :param freq: List of frequencies in float
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
            flux_single = flux[i] * (freq[0] / ref_freq[i]) ** spec_indx[i]
            flux_array = np.array([[flux_single]])
        else:
            if ref_freq[i] > 0:
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


def plot_errors(
    orig,
    comp,
    input_image,
    match_sep,
    sources_in_file=None,
    sources_out_file=None,
    flux_limit=0.0,
    plot_file=None,
):
    """
    Plot the position and flux errors for source input and output

    :param orig: Input source list in skycomponent format
    :param comp: Output source list in skycomponent format
    :param input_image: Input image for Gaussian fits
    :param match_sep: The criteria for maximum separation
    :param sources_in_file: Name of input source file to be read (only use this for spectral index)
    :param sources_out_file: Name of csv file to be read (for sources out)
    :param flux_limit: The flux criterion for plotting cutoff
    :param plot_file: prefix of the plot files
    :return

    """
    log.info("Plotting skycomponents to check the accuracy of the source finder.")
    log.info("Use flux cutoff:{}".format(flux_limit))

    image = import_image_from_fits(input_image, fixpol=True)

    img_size = np.rad2deg(image.image_acc.wcs.wcs.cdelt[1])
    phasecentre = image.image_acc.phasecentre
    nchan = image["pixels"].shape[0]
    refchan = nchan // 2

    # If reading spectral index from files
    if sources_in_file is not None and ".txt" in sources_in_file:

        # Currently only applied to txt files
        log.info("Reading spectral index from sources in file.")
        data_in = np.loadtxt(sources_in_file, delimiter=",", unpack=True)
        indexes_in = data_in[7]

    else:
        log.info("Using fitted spectral index in.")
        indexes_in = None

    if sources_out_file is not None:
        log.info("Reading spectral index from sources out file.")
        try:
            data = pd.read_csv(sources_out_file, engine="python")
            indexes_out = data["Spectral index"].to_numpy()
        except KeyError:
            log.warning(
                "File does not contain spectral index information, using fitted values instead."
            )
            indexes_out = None
    else:
        log.info("Using fitted spectral index out.")
        indexes_out = None

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

    log.info("Plotting spectral index.")
    if nchan > 1 or indexes_out is not None:

        spec_in, spec_out = plot_multifreq_spectral_index(
            comp,
            orig,
            phasecentre,
            plot_file=plot_file,
            tol=match_sep,
            flux_limit=flux_limit,
            spec_indx_test=indexes_out,
            spec_indx_ref=indexes_in,
            plot_diagnostics=True,
        )

    log.info("Plotting done.")

    return


if __name__ == "__main__":

    # Get command line inputs
    parser = cli_parser()
    args = parser.parse_args()
    analyze_image(args)
