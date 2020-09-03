""" Visibility operations

"""

__all__ = ['convert_blockvisibility_to_xvisibility',
           'convert_xvisibility_to_blockvisibility']

import logging

import numpy
import pandas
from astropy.coordinates import SkyCoord
from astropy.time import Time

from rascil.data_models.memory_data_models import Configuration
from rascil.data_models.memory_data_models import BlockVisibility
from rascil.data_models.memory_data_models_xarray import XVisibility
from rascil.data_models.polarisation import PolarisationFrame, correlate_polarisation
from rascil.processing_components.util import hadec_to_azel
from rascil.processing_components.util.geometry import calculate_transit_time
from rascil.processing_components.util.uvw_coordinates import uvw_ha_dec

log = logging.getLogger('logger')

def convert_blockvisibility_to_xvisibility(vis: BlockVisibility) -> XVisibility:
    """Convert visibility to XVisibility

    :param vis: BlockVisibility format
    :return: XVisibility
    """
    ntimes, nants, _, nchan, npol = vis.vis.shape

    def gen_base(nant):
        for ant1 in range(1, nant):
            for ant2 in range(ant1):
                yield ant1, ant2

    nant = 5
    baselines = pandas.MultiIndex.from_tuples(gen_base(nant), names=('antenna1', 'antenna2'))
    nbaselines = len(baselines)
    
    def upper_triangle(x):
        x_reshaped = numpy.zeros([ntimes, nbaselines, nchan, npol], dtype=x.dtype)
        for itime, _ in enumerate(vis.time):
            for ibaseline, baseline in enumerate(baselines):
                ant1 = baseline[0]
                ant2 = baseline[1]
                for chan, freq in enumerate(vis.frequency):
                    for ipol, pol in enumerate(vis.polarisation_frame.names):
                        x_reshaped[itime, ibaseline, chan, ipol] = \
                            x[itime, ant2, ant1, chan, ipol]
        return x_reshaped
                    
    def uvw_reshape(uvw):
        uvw_reshaped = numpy.zeros([ntimes, nbaselines, 3])
        for itime, _ in enumerate(vis.time):
            for ibaseline, baseline in enumerate(baselines):
                ant1 = baseline[0]
                ant2 = baseline[1]
                uvw_reshaped[itime, ibaseline, :] = uvw[itime, ant2, ant1]
        return uvw_reshaped

    return XVisibility(frequency=vis.frequency,
                       channel_bandwidth=vis.channel_bandwidth,
                       polarisation_frame=vis.polarisation_frame,
                       phasecentre=vis.phasecentre,
                       configuration=vis.configuration,
                       time=vis.time,
                       baselines=baselines,
                       uvw=uvw_reshape(vis.uvw),
                       vis=upper_triangle(vis.vis),
                       flags=upper_triangle(vis.flags),
                       weight=upper_triangle(vis.weight),
                       integration_time=vis.integration_time)


def convert_xvisibility_to_blockvisibility(xvis: XVisibility) -> BlockVisibility:
    """Convert xvisibility to blockisibility

    :param vis:
    :param othervis:
    :return: Visibility vis
    """
    return BlockVisibility(frequency=xvis.frequency,
                      channel_bandwidth=xvis.channel_bandwidth,
                      polarisation_frame=xvis.polarisation_frame,
                      phasecentre=xvis.phasecentre,
                      configuration=xvis.configuration,
                      time=xvis.time,
                      vis=xvis.vis,
                      flags=xvis.flags,
                      weight=xvis.weight,
                      integration_time=xvis.integration_time)
