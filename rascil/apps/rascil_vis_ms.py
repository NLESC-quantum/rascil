""" RASCIL MS visualisaation

"""

import argparse
import logging
import os
import sys

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from rascil.processing_components import create_blockvisibility_from_ms, plot_configuration, \
    plot_visibility, plot_uvcoverage

from rascil.apps.common import display_ms_as_image

from rascil.apps.apps_parser import apps_parser_ingest, apps_parser_app

log = logging.getLogger("rascil-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


def cli_parser():
    """ Get a command line parser and populate it with arguments

    First a CLI argument parser is created. Each function call adds more arguments to the parser.

    :return: CLI parser argparse
    """
    
    parser = argparse.ArgumentParser(description='RASCIL ms visualisation')
    parser.add_argument('--ingest_msname', type=str, default=None, help='MeasurementSet to be read')
    parser.add_argument('--logfile', type=str, default=None,
                        help='Name of logfile (default is to construct one from msname)')

    return parser


def visualise(args):
    """ MS Visualiser

    Performs simple visualisations of the MS

    :param args: argparse with appropriate arguments
    :return: None
    """
    
    # We need to tell all the Dask workers to use the same log
    cwd = os.getcwd()
    
    assert args.ingest_msname is not None, "Input msname must be specified"
    
    if args.logfile is None:
        logfile = args.ingest_msname.replace('.ms', ".log")
    else:
        logfile = args.logfile
    
    def init_logging():
        logging.basicConfig(filename=logfile,
                            filemode='a',
                            format='%(asctime)s.%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%d/%m/%Y %I:%M:%S %p',
                            level=logging.INFO)
    
    init_logging()
    
    log.info("\nRASCIL MS Visualiser\n")
    
    display_ms_as_image(msname=args.ingest_msname)
    
    bvis_list = create_blockvisibility_from_ms(args.ingest_msname)
    
    log.info(bvis_list[0])
    
    plt.clf()
    plot_configuration(bvis_list[0].configuration)
    plt.savefig(args.ingest_msname.replace('.ms', "_configuration.png"))
    
    plt.clf()
    plot_uvcoverage(bvis_list)
    plt.savefig(args.ingest_msname.replace('.ms', "_uvcoverage.png"))
    plt.clf()
    
    nchan = bvis_list[0]["vis"].shape[-2]
    
    plt.clf()
    plot_visibility(bvis_list, plot_file=args.ingest_msname.replace('.ms', "_visibility_amp.png"), chan=nchan // 2)
    
    plt.clf()
    plot_visibility(bvis_list, plot_file=args.ingest_msname.replace('.ms', "_visibility_phase.png"), chan=nchan // 2,
                    y="phase")


def main():
    # Get command line inputs
    parser = cli_parser()
    args = parser.parse_args()
    visualise(args)


if __name__ == "__main__":
    main()
