""" Visibility operations

"""
import xarray

__all__ = ['convert_visibility_to_xvisibility',
           'convert_xvisibility_to_visibility']

import logging
from typing import Union, List

import numpy 

from rascil.data_models.memory_data_models import Visibility, QA
from rascil.data_models.polarisation import PolarisationFrame

log = logging.getLogger('logger')

def convert_visibility_to_xvisibility(vis: Visibility) -> xarray.Dataset:
    """Convert visibility to XVisibility

    :param bvis:
    :param othervis:
    :return: XVisibility vis
    """
    coords = {"time": vis.time,
              "polarisation": vis.polarisation_frame.names,
              "location": numpy.zeros(3)}
    
    xvis_dict = {}
    xvis_dict["data"] = xarray.DataArray(vis.vis, dims=["time", "polarisation"])
    xvis_dict["uvw"] = xarray.DataArray(vis.uvw, dims=["time", "location"])
    xvis_dict["weight"] = xarray.DataArray(vis.weight, dims=["time", "polarisation"])
    xvis_dict["imaging_weight"] = xarray.DataArray(vis.imaging_weight,
                                                   dims=["time", "polarisation"])
    xvis_dict["flags"] = xarray.DataArray(vis.flags, dims=["time", "polarisation"])
    xvis_dict["frequency"] = xarray.DataArray(vis.frequency, dims=["time"])
    xvis_dict["channel_bandwidth"] = xarray.DataArray(vis.channel_bandwidth, dims=["time"])
    xvis_dict["antenna1"] = xarray.DataArray(vis.antenna1, dims=["time"])
    xvis_dict["antenna2"] = xarray.DataArray(vis.antenna2, dims=["time"])
    xvis_dict["integration_time"] = xarray.DataArray(vis.integration_time, dims=["time"])
    xvis_xds = xarray.Dataset(xvis_dict, coords=coords)
    xvis_xds.attrs['source'] = vis.source
    xvis_xds.attrs['meta'] = vis.meta
    
    return xvis_xds
    # xvis_xds.attrs['xvis_name'] = xvis_name
    # xvis_xds.attrs['xvis_long'] = xvis_location['m0']['value']
    # xvis_xds.attrs['xvis_lat'] = xvis_location['m1']['value']
    # xvis_xds.attrs['xvis_elevation'] = xvis_location['m2']['value']
    # xvis_xds.attrs['long_units'] = xvis_location['m0']['unit']
    # xvis_xds.attrs['lat_units'] = xvis_location['m1']['unit']
    # xvis_xds.attrs['elevation_units'] = xvis_location['m2']['unit']
    # xvis_xds.attrs['coordinate_system'] = xvis_location['refer']


def convert_xvisibility_to_visibility(xvis: xarray.Dataset) \
        -> Visibility:
    """Convert xvisibility to blockisibility

    :param vis:
    :param othervis:
    :return: Visibility vis
    """
    return Visibility()

