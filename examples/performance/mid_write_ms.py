"""Time creation and writing of MeasurementSet

"""
import logging
import pprint
import sys

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models import PolarisationFrame, export_blockvisibility_to_hdf5
from rascil.processing_components import create_named_configuration
from rascil.processing_components.visibility import create_blockvisibility
from rascil.processing_components.visibility.base import export_blockvisibility_to_ms

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


def simulation(args):
    pp = pprint.PrettyPrinter()
    pp.pprint(vars(args))
    
    ra = args.ra
    declination = args.declination
    band = args.band
    
    if args.duration == "short":
        integration_time = 1.0
        time_range = [-180.0 / 3600.0, 180.0 / 3600.0]
    elif args.duration == "medium":
        integration_time = 10.0
        time_range = [-0.5, 0.5]
    elif args.duration == "long":
        integration_time = 10.0
        time_range = [-4.0, 4.0]
    else:
        args.duration = "custom"
        integration_time = args.integration_time
        time_range = args.time_range
    
    image_polarisation_frame = PolarisationFrame(args.image_pol)
    vis_polarisation_frame = PolarisationFrame(args.vis_pol)
    log.info("Image polarisation: {}".format(str(image_polarisation_frame)))
    log.info("Vis polarisation: {}".format(str(vis_polarisation_frame)))
    
    log.info("Simulating {duration} observation: {time_range} hours in integrations of {integration_time}s"
             .format(duration=args.duration, time_range=time_range, integration_time=integration_time))
    
    rmax = args.rmax
    
    # Set up details of simulated observation
    nchan = args.nchan
    channel_width = args.channel_width
    if band == 'B1':
        if channel_width is None:
            channel_width = 1.0e8 / nchan
        frequency = 0.7650e9 + numpy.arange(nchan) * channel_width
    elif band == 'B1LOW':
        if channel_width is None:
            channel_width = 1.0e8 / nchan
        frequency = 0.350e9 + numpy.arange(nchan) * channel_width
    elif band == 'B2':
        if channel_width is None:
            channel_width = 1.0e8 / nchan
        frequency = 1.36e9 + numpy.arange(nchan) * channel_width
    elif band == 'Ku':
        if channel_width is None:
            channel_width = 1.0e9 / nchan
        frequency = 12.179e9 + numpy.arange(nchan) * channel_width
    else:
        raise ValueError("Unknown band %s" % band)
    
    frequency = numpy.array(frequency)
    channel_bandwidth = numpy.repeat(channel_width, len(frequency))
    
    phasecentre = SkyCoord(ra=ra * u.deg, dec=declination * u.deg, frame='icrs', equinox='J2000')
    
    mid = create_named_configuration(args.configuration, rmax=rmax)
    
    times = numpy.arange(time_range[0] * 3600, time_range[1] * 3600, integration_time)
    times *= numpy.pi / 43200.0
    
    import time
    start_time = time.time()
    entire_bvis = create_blockvisibility(mid, times, frequency=frequency,
                                         channel_bandwidth=channel_bandwidth,
                                         weight=1.0,
                                         phasecentre=phasecentre,
                                         polarisation_frame=vis_polarisation_frame)
    ms_createtime = time.time() - start_time
    log.info("Time to create BV = {:.3f} seconds".format(ms_createtime))
    
    print("Size of BlockVisibility = {:.3f} GB".format(entire_bvis.size()))
    
    msfile = "ms_write.ms"
    log.info("Writing BlockVisibility to MS")
    export_blockvisibility_to_ms(msfile, [entire_bvis])
    ms_writetime = time.time() - start_time
    log.info("Time to write BlockVisibility as MS = {:.3f} seconds".format(ms_writetime))

    log.info("Writing BlockVisibility to HDF5")
    start_time = time.time()
    hdf5file = "ms_write.hdf5"
    export_blockvisibility_to_hdf5([entire_bvis], hdf5file)
    hdf5_writetime = time.time() - start_time
    log.info("Time to write BlockVisibility as HDF5 = {:.3f} seconds".format(hdf5_writetime))

    return True


def cli_parser():
    global parser
    import argparse
    
    parser = argparse.ArgumentParser(description='Simulate SKA-MID direction dependent errors')
    parser.add_argument('--ra', type=float, default=0.0, help='Right ascension (degrees)')
    parser.add_argument('--declination', type=float, default=-40.0, help='Declination (degrees)')
    parser.add_argument('--rmax', type=float, default=2e5, help='Maximum distance of station from centre (m)')
    parser.add_argument('--band', type=str, default='B2', help="Band")
    parser.add_argument('--configuration', type=str, default='MID', help="Configuration: MID | MEERKAT+")
    parser.add_argument('--nchan', type=int, default=1, help="Number of frequency channels")
    parser.add_argument('--channel_width', type=float, default=None, help='Channel bandwidth (Hz)')
    parser.add_argument('--integration_time', type=float, default=180, help='Integration time (s)')
    parser.add_argument('--time_range', type=float, nargs=2, default=[-4.0, 4.0], help='Time range in hour angle')
    parser.add_argument('--image_pol', type=str, default='stokesIQUV', help='RASCIL polarisation frame for image')
    parser.add_argument('--vis_pol', type=str, default='linear', help='RASCIL polarisation frame for visibility')
    parser.add_argument('--duration', type=str, default='long', help="Type of duration: long or medium or short")
    
    return parser


if __name__ == "__main__":
    # Get command line inputs
    
    # Get command line inputs
    parser = cli_parser()
    
    args = parser.parse_args()
    
    simulation(args)
