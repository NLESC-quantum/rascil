""" Visibility operations

"""
import xarray

__all__ = ['convert_visibility_to_xvisibility',
           'convert_xvisibility_to_visibility']

import logging

from rascil.data_models.memory_data_models import Visibility
from rascil.data_models.memory_data_models_xarray import XVisibility
from rascil.data_models.polarisation import PolarisationFrame

log = logging.getLogger('logger')

def convert_visibility_to_xvisibility(vis: Visibility) -> XVisibility:
    """Convert visibility to XVisibility

    :param vis: Visibility format
    :return: XVisibility
    """
    return XVisibility(frequency=vis.frequency,
                       channel_bandwidth=vis.channel_bandwidth,
                       polarisation_frame=vis.polarisation_frame,
                       phasecentre=vis.phasecentre,
                       configuration=vis.configuration,
                       time=vis.time,
                       antenna1=vis.antenna1,
                       antenna2=vis.antenna2,
                       vis=vis.vis,
                       flags=vis.flags,
                       weight=vis.weight,
                       integration_time=vis.integration_time)


def convert_xvisibility_to_visibility(xvis: XVisibility) -> Visibility:
    """Convert xvisibility to blockisibility

    :param vis:
    :param othervis:
    :return: Visibility vis
    """
    return Visibility(frequency=xvis.frequency,
                       channel_bandwidth=xvis.channel_bandwidth,
                       polarisation_frame=xvis.polarisation_frame,
                       phasecentre=xvis.phasecentre,
                       configuration=xvis.configuration,
                       time=xvis.time,
                       antenna1=xvis.antenna1,
                       antenna2=xvis.antenna2,
                       vis=xvis.vis,
                       flags=xvis.flags,
                       weight=xvis.weight,
                       integration_time=xvis.integration_time)

