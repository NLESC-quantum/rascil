""" RASCIL MS advice

"""

import argparse
import logging
import sys
import pprint

from rascil.processing_components import create_blockvisibility_from_ms, advise_wide_field

log = logging.getLogger("rascil-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


def cli_parser():
    """ Get a command line parser and populate it with arguments

    First a CLI argument parser is created. Each function call adds more arguments to the parser.

    :return: CLI parser argparse
    """
    
    parser = argparse.ArgumentParser(description='RASCIL imaging advise')
    parser.add_argument('--ingest_msname', type=str, default=None, help='MeasurementSet to be read')
    parser.add_argument('--ingest_dd', type=int, nargs="*", default=[0],
                        help='Data descriptors in MS to read (all must have the same number of channels)')
    parser.add_argument('--logfile', type=str, default=None,
                        help='Name of logfile (default is to construct one from msname)')
    parser.add_argument('--guard_band_image', type=float, default=3.0,
                        help="Size of field of view in primary beams")
    parser.add_argument('--oversampling_synthesised_beam', type=float, default=3,
                        help='Pixels per syntheised beam')
    parser.add_argument('--dela', type=float, default=0.02,
                        help='Maximum allowed decorrelation')
    
    return parser


def advise(args):
    """ MS Advicer

    Delivers imaging advice for an MS

    :param args: argparse with appropriate arguments
    :return: None
    """
    
    assert args.ingest_msname is not None, "Input msname must be specified"
    
    if args.logfile is None:
        logfile = args.ingest_msname.replace('.ms', "_advise.log")
    else:
        logfile = args.logfile
    
    def init_logging():
        logging.basicConfig(filename=logfile,
                            filemode='a',
                            format='%(asctime)s.%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%d/%m/%Y %I:%M:%S %p',
                            level=logging.INFO)
    
    init_logging()
    
    log.info("\nRASCIL MS Imaging Advice\n")
    
    log.info(pprint.pformat(vars(args)))
    
    bvis_list = create_blockvisibility_from_ms(args.ingest_msname, selected_dds=args.ingest_dd)
    
    log.info(f"MS loaded into BlockVisibility:\n{bvis_list[0]}\n")
    
    return advise_wide_field(bvis_list[0],
                             guard_band_image=args.guard_band_image,
                             oversampling_synthesised_beam=args.oversampling_synthesised_beam,
                             delA=args.dela, verbose=True)


if __name__ == "__main__":
    parser = cli_parser()
    args = parser.parse_args()
    advise(args)
