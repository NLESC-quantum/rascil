"""
Stand-alone application for finding sources with PyBDSF
"""

import argparse
import datetime
import logging
import os
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import pandas as pd
import bdsf
import astropy.units as u
from astropy.coordinates import SkyCoord
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import create_low_test_skycomponents_from_gleam
from rascil.processing_components.skycomponent.operations import create_skycomponent, find_skycomponent_matches, \
    apply_beam_to_skycomponent
from rascil.processing_components.image.operations import import_image_from_fits
from rascil.processing_components.imaging.primary_beams import create_pb

log = logging.getLogger("rascil-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


def cli_parser():
    """ Get a command line parser and populate it with arguments

    :param parser: argparse    
    :return: CLI parser argparse
    """
    parser = argparse.ArgumentParser(description='RASCIL continuum image checker')
    parser.add_argument('--ingest_fitsname', type=str, default=None, help='FITS file to be read')
    parser.add_argument('--finder_bmaj', type=float, default=1.0,
                        help='Major axis of the restoring beam')
    parser.add_argument('--finder_bmin', type=float, default=1.0,
                        help='Minor axis of the restoring beam')
    parser.add_argument('--finder_pos_angle', type=float, default=0.0,
                        help='Positioning angle of the restoring beam')
    parser.add_argument('--finder_th_isl', type=float, default=5.0,
                        help='Threshold to determine the size of the islands')
    parser.add_argument('--finder_th_pix', type=float, default=10.0,
                        help='Threshold to detect source (peak value)')
    parser.add_argument('--apply_primary', type=str, default='True',
                        help='Whether to apply primary beams')
    parser.add_argument('--telescope_model', type=str, default='MID',
                        help='The telescope to generate primary beam correction')
    parser.add_argument('--match_sep', type=float, default=1.e-5,
                        help='Maximum separation in radians for the source matching')
    parser.add_argument('--source_file', type=str, default='output.csv',
                        help='Name of output source file')
    parser.add_argument('--logfile', type=str, default=None,
                        help='Name of output log file')
    
    return parser


def analyze_image(args):
    """
    Main analyais routine.
    
    :param args: argparse with appropriate arguments

    :return:    A list of sources and their matches to original sources

    """
    
    cwd = os.getcwd()
    
    assert args.ingest_fitsname is not None, "Input FITS file name must be specified"
    
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
    print(im)
    
    bmaj = args.finder_bmaj
    bmin = args.finder_bmin
    pos_angle = args.finder_pos_angle
    beam_info = (bmaj, bmin, pos_angle)
    
    th_isl = args.finder_th_isl
    th_pix = args.finder_th_pix
    
    # Adjust the best restoring beam configuration-- not sure about this yet
    # factor = 1/(npixels/512) **2
    # beam_info = (beam_info[0]* factor, beam_info[1] * factor, beam_info[2])
    
    freq = im.frequency.data[0]
    log.info("Use restoring beam: {}".format(beam_info))
    log.info("Use threshold: {}".format(th_isl, th_pix))
    
    ci_checker(input_image, beam_info, source_file, th_isl, th_pix)
    out = create_source_to_skycomponent(source_file, freq)
    
    if args.apply_primary:
        telescope = args.telescope_model
        out = add_primary_beams(input_image, out, telescope)
    
    tol_val = args.match_sep
    results = check_source(im.image_acc.phasecentre, freq, out, tol_val)
    
    log.info("Resulting list of items {}".format(results))
    
    log.info("Started  : {}".format(starttime))
    log.info("Finished : {}".format(datetime.datetime.now()))
    
    return results


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
    
    return


def create_source_to_skycomponent(source_file, freq):
    """
    Put the sources into RASCIL-readable skycomponents

    :param source_file: Output file name of the source list
    :param freq: Frequency or list of frequencies

    :return comp: List of skycomponents
    """
    
    data = pd.read_csv(source_file, sep=r'\s*,\s*', skiprows=5)
    comp = []
    for i in range(len(data.index)):
        direc = SkyCoord(ra=data['RA'][i] * u.deg, dec=data['DEC'][i] * u.deg, frame='icrs', equinox='J2000')
        f = data['Total_flux'][i]
        comp.append(create_skycomponent(direction=direc, flux=np.array([[f]]), frequency=np.array([freq]),
                                        polarisation_frame=PolarisationFrame('stokesI')))
    
    return comp


def add_primary_beams(input_image, comp, telescope):
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


def check_source(centre, freq, comp, tol_val):
    """
    Check the difference between output sources and input.
    
    :param centre: Phase centre of image
    :param freq: Frequency or list of frequencies.
    :param comp: Output source list in skycomponent format
    :param tol_val: The criteria for maximum separation
 
    :return matches: List of matched skycomponents
 
    """
    orig = create_low_test_skycomponents_from_gleam(flux_limit=1.0, phasecentre=centre,
                                                    frequency=np.array([freq]),
                                                    polarisation_frame=PolarisationFrame('stokesI'), radius=0.5)
    
    # separations = find_separation_skycomponents(comp, orig)
    matches = find_skycomponent_matches(comp, orig, tol=tol_val)
    
    return matches

if __name__ == "__main__":

    # Get command line inputs
    parser = cli_parser()
    args = parser.parse_args()
    analyze_image(args)

