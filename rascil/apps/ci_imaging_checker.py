"""
Stand-alone application for finding sources with PyBDSF
"""

import argparse
import datetime
import logging
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')
import pandas as pd
import bdsf
import astropy.units as u
from astropy import modeling
from astropy.coordinates import SkyCoord
from rascil.data_models import PolarisationFrame, import_skycomponent_from_hdf5
from rascil.processing_components import create_low_test_skycomponents_from_gleam
from rascil.processing_components.skycomponent.operations import create_skycomponent, find_skycomponent_matches, \
    apply_beam_to_skycomponent
from rascil.processing_components.image.operations import import_image_from_fits
from rascil.processing_components.imaging.primary_beams import create_pb


class FileNotFoundError(Exception):
    pass


class FileFormatError(Exception):
    pass


log = logging.getLogger("rascil-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


def cli_parser():
    """ Get a command line parser and populate it with arguments

    :param parser: argparse
    :return: CLI parser argparse
    """
    parser = argparse.ArgumentParser(description='RASCIL continuum imaging checker')
    parser.add_argument('--ingest_fitsname', type=str, default=None, help='FITS file to be read')
    parser.add_argument('--finder_beam_maj', type=float, default=1.0,
                        help='Major axis of the restoring beam')
    parser.add_argument('--finder_beam_min', type=float, default=1.0,
                        help='Minor axis of the restoring beam')
    parser.add_argument('--finder_beam_pos_angle', type=float, default=0.0,
                        help='Positioning angle of the restoring beam')
    parser.add_argument('--finder_th_isl', type=float, default=5.0,
                        help='Threshold to determine the size of the islands')
    parser.add_argument('--finder_th_pix', type=float, default=10.0,
                        help='Threshold to detect source (peak value)')
    parser.add_argument('--apply_primary', type=str, default='True',
                        help='Whether to apply primary beam')
    parser.add_argument('--telescope_model', type=str, default='MID',
                        help='The telescope to generate primary beam correction')
    parser.add_argument('--check_source', type=str, default=False,
                        help='Option to check with original input source catalogue')
    parser.add_argument('--input_source_format', type=str, default='external',
                        help='The input format of the source catalogue')
    parser.add_argument('--input_source_filename', type=str, default=None,
                        help='If use external source file, the file name of source file')
    parser.add_argument('--match_sep', type=float, default=1.e-5,
                        help='Maximum separation in radians for the source matching')
    parser.add_argument('--source_file', type=str, default=None,
                        help='Name of output source file')
    parser.add_argument('--logfile', type=str, default=None,
                        help='Name of output log file')

    return parser


def analyze_image(args):
    """
    Main analysis routine.

    :param args: argparse with appropriate arguments

    :return:    A list of sources and their matches to original sources

    """

    cwd = os.getcwd()

    if args.ingest_fitsname is None:
        raise FileNotFoundError("Input FITS file name must be specified")

    if args.logfile is None:
        logfile = args.ingest_fitsname.replace('.fits', ".log")
    else:
        logfile = args.logfile

    if args.source_file is None:
        source_file = args.ingest_fitsname.replace('.fits', ".csv")
    else:
        source_file = args.source_file

    def init_logging():
        logging.basicConfig(filename=logfile,
                            filemode='a',
                            format='%(asctime)s.%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%d/%m/%Y %I:%M:%S %p',
                            level=logging.INFO)

    init_logging()

    log.info("\nRASCIL Continuum Imagine Checker\n")

    starttime = datetime.datetime.now()
    log.info("Started : {}".format(starttime))
    log.info("Writing log to {}".format(logfile))

    input_image = args.ingest_fitsname

    im = import_image_from_fits(args.ingest_fitsname)

    beam_maj = args.finder_beam_maj
    beam_min = args.finder_beam_min
    beam_pos_angle = args.finder_beam_pos_angle
    beam_info = (beam_maj, beam_min, beam_pos_angle)

    th_isl = args.finder_th_isl
    th_pix = args.finder_th_pix


    freq = im.frequency.data[0]
    log.info("Use restoring beam: {}".format(beam_info))
    log.info("Use threshold: {}, {}".format(th_isl, th_pix))

    ci_checker(input_image, beam_info, source_file, th_isl, th_pix)
    out = create_source_to_skycomponent(source_file, freq)

    if args.apply_primary:
        telescope = args.telescope_model
        out = add_primary_beam(input_image, out, telescope)

    if args.check_source:

        match_sep = args.match_sep
        if args.input_source_format == 'external':
            if '.h5' in args.input_source_filename or 'hdf' in args.input_source_filename:
                orig = import_skycomponent_from_hdf5(args.input_source_filename)

            elif '.txt' in args.input_source_filename:
                orig = read_skycomponent_from_txt(args.input_source_filename, freq)
            else:
                raise FileFormatError("Input file must be of format: hdf5 or txt.")
        else: # Use internally provided GLEAM model
            orig = create_low_test_skycomponents_from_gleam(flux_limit=1.0, phasecentre=im.image_acc.phasecentre,
                                                            frequency=np.array([freq]),
                                                            polarisation_frame=PolarisationFrame('stokesI'), radius=0.5)

        results = check_source(orig, out, match_sep)
        log.info("Resulting list of matched items {}".format(results))

    else:
        results = None


    log.info("Started  : {}".format(starttime))
    log.info("Finished : {}".format(datetime.datetime.now()))

    return out, results


def bdsf_qa_image(im_data):
    """Assess the quality of an image

    Set of statistics of an image: max, min, maxabs, rms, sum, medianabs,
    medianabsdevmedian, median and mean.

    :param im_data: image data
    :return: None
    """

    image_stats = {
        'shape': str(im_data.data.shape),
        'max': np.max(im_data),
        'min': np.min(im_data),
        'maxabs': np.max(np.abs(im_data)),
        'rms': np.std(im_data),
        'sum': np.sum(im_data),
        'medianabs': np.median(np.abs(im_data)),
        'medianabsdevmedian': np.median(np.abs(im_data - np.median(im_data))),
        'median': np.median(im_data),
        'mean': np.mean(im_data)
    }

    log.info("QA of image:")
    for item in image_stats.items():
        log.info("    {}".format(item))

    return image_stats


def ci_checker_diagnostics(bdsf_image, input_image):
    """
    Log and plot diagnostics of an image generated by bdfs. Currently only
    the residual is analysed and a histogram of the pixel counts produced with
    mean, RMS and Gaussian fit.

    :param bdsf_image: pybdsf image object
    :param input_image: file name of input image

    :return None
    """

    im_data = bdsf_image.resid_gaus_arr
    image_stats = bdsf_qa_image(im_data)

    fig, ax = plt.subplots()

    counts, bins, _ = ax.hist(im_data.ravel(), bins=1000, density=False,
                              zorder=5, histtype="step")

    fitter = modeling.fitting.LevMarLSQFitter()
    model = modeling.models.Gaussian1D(
        amplitude=image_stats["max"],
        mean=image_stats["mean"],
        stddev=image_stats["rms"]
    )

    # "bins" are the bin edge points, so need the mid points.
    mid_points = bins[:-1] + 0.5*abs(bins[1:] - bins[:-1])

    fitted_model = fitter(model, mid_points, counts)

    ax.plot(mid_points, fitted_model(mid_points), label="Gaussian", zorder=10)
    ax.axvline(image_stats["mean"], color="C2",
               linestyle="--", label="mean", zorder=15)

    # Add shaded region of withth 2*RMS centered on the mean.
    rms_region = [image_stats["mean"] - image_stats["rms"],
                  image_stats["mean"] + image_stats["rms"]]
    ax.axvspan(rms_region[0], rms_region[1], facecolor="C2",
               alpha=0.3, zorder=10, label="RMS")

    ax.set_yscale("symlog")
    ax.set_ylabel("Counts")
    ax.set_xlabel("Value")
    ax.legend()

    # Create histogram file name from input image name removeing file extension.
    save_hist = input_image.replace(
        '.fits' if '.fits' in input_image else '.h5', "") \
        + "_resid_gaus_hist.png"

    plt.tight_layout()

    log.info('Saving histogram to "{}"'.format(save_hist))
    plt.savefig(save_hist)
    plt.close()

    return


def ci_checker(input_image, beam_info, source_file, th_isl, th_pix):
    """
    PyBDSF-based source finder

    :param input_image
    :param beam_info : Size of restoring beam as (bmaj, bmin, pos_angle)
    :param source_file : Output file name of the source list
    :param th_isl : Island threshold
    :param th_pix: Peak threshold

    : return None
    """

    # Process image.
    img = bdsf.process_image(input_image, beam=beam_info, thresh_isl=th_isl, thresh_pix=th_pix)

    # Write the source catalog and the residual image.
    img.write_catalog(outfile=source_file, format='csv', catalog_type='srl', clobber=True)
    img.write_catalog(format='fits', catalog_type='srl', clobber=True)
    img.export_image(img_type='gaus_resid', clobber=True)
    # img.export_image(img_type='gaus_model', clobber=True)

    ci_checker_diagnostics(img, input_image)

    return


def create_source_to_skycomponent(source_file, freq):
    """
    Put the sources into RASCIL-readable skycomponents

    :param source_file: Output file name of the source list
    :param freq: Frequency or list of frequencies

    :return comp: List of skycomponents
    """

    data = pd.read_csv(source_file, sep=r'\s*,\s*', skiprows=5, engine='python')
    comp = []
    for i, row in data.iterrows():

        direc = SkyCoord(ra=row['RA'] * u.deg, dec=row['DEC'] * u.deg, frame='icrs', equinox='J2000')
        f = row['Total_flux']
        if f > 0: # filter out ghost sources
            comp.append(create_skycomponent(direction=direc, flux=np.array([[f]]), frequency=np.array([freq]),
                                            polarisation_frame=PolarisationFrame('stokesI')))

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

    # separations = find_separation_skycomponents(comp, orig)
    matches = find_skycomponent_matches(comp, orig, tol=match_sep)

    return matches

def read_skycomponent_from_txt(filename, freq):
    """
    Read source input from a txt file and make the date into skycomponents

    :param filename: Name of input file
    :param freq: Frequency or list of frequencies
    :return comp: List of skycomponents
    """

    data = np.loadtxt(filename, delimiter=',', unpack=True)
    comp = []

    ra = data[0]
    dec = data[1]
    flux = data[2]

    for i, row in enumerate(ra):

        direc = SkyCoord(ra=ra[i] * u.deg, dec=dec[i] * u.deg, frame='icrs', equinox='J2000')
        comp.append(create_skycomponent(direction=direc, flux=np.array([[flux[i]]]), frequency=np.array([freq]),
                                	polarisation_frame=PolarisationFrame('stokesI')))

    return comp

if __name__ == "__main__":

    # Get command line inputs
    parser = cli_parser()
    args = parser.parse_args()
    analyze_image(args)
