""" Visibility operations

"""

__all__ = ['create_ximage',
           'convert_image_to_ximage']

import logging
import dask
import xarray

import numpy

from rascil.data_models import Image, PolarisationFrame

log = logging.getLogger('logger')

def create_ximage(axes, cellsize, phasecentre, frequency, polarisation_frame=None,
                  dtype=None, data=None, wcs=None):
    """ Create an image as an XDataset
    
    :param axes:
    :param cellsize:
    :param frequency:
    :param phasecentre:
    :param polarisation_frame:
    :return: XImage
    """
    
    nx, ny = axes
    cx = phasecentre.ra.to("deg").value
    cy = phasecentre.dec.to("deg").value
    cellsize = numpy.rad2deg(cellsize)
    
    dims = ["frequency", "polarisation", "lat", "lon"]
    coords = {"frequency": frequency,
              "polarisation": polarisation_frame.names,
              "lat": numpy.linspace(cy - cellsize * ny // 2, cy + cellsize * ny // 2, ny),
              "lon": numpy.linspace(cx - cellsize * nx // 2, cx + cellsize * nx // 2, nx)
              }
        
    attrs = dict()
    attrs["phasecentre"] = phasecentre
    attrs["polarisation_frame"] = polarisation_frame
    ramesh, decmesh = numpy.meshgrid(numpy.arange(ny), numpy.arange(nx))
    attrs["ra"], attrs["dec"] = wcs.sub([1, 2]).wcs_pix2world(ramesh, decmesh, 0)

    nchan = len(frequency)
    npol = polarisation_frame.npol
    if dtype is None:
        dtype = "float"

    if data is None:
        data = numpy.zeros([nchan, npol, ny, nx], dtype=dtype)
    else:
        assert data.shape == (nchan, npol, ny, nx), data.shape
        
    return xarray.DataArray(data, dims=dims, coords=coords, attrs=attrs)


def convert_image_to_ximage(im: Image):
    """ Convert an Image to an XImage

    :param axes:
    :param cellsize:
    :param frequency:
    :param phasecentre:
    :param polarisation_frame:
    :return: XImage
    """
    
    phasecentre = im.phasecentre
    nchan, npol, ny, nx = im.shape
    cx = phasecentre.ra.to("deg").value
    cy = phasecentre.dec.to("deg").value
    cellsize = numpy.abs(im.wcs.wcs.cdelt[1])
    frequency = im.frequency
    polarisation_frame = im.polarisation_frame
    
    dims = ["frequency", "polarisation", "lat", "lon"]
    coords = {"frequency": frequency,
              "polarisation": polarisation_frame.names,
              "lat": numpy.linspace(cy - cellsize * ny // 2, cy + cellsize * ny // 2, ny),
              "lon": numpy.linspace(cx - cellsize * nx // 2, cx + cellsize * nx // 2, nx)
              }
    
    attrs = dict()
    attrs["phasecentre"] = phasecentre
    attrs["polarisation_frame"] = polarisation_frame
    ramesh, decmesh = numpy.meshgrid(numpy.arange(ny), numpy.arange(nx))
    attrs["ra"], attrs["dec"] = im.wcs.sub([1, 2]).wcs_pix2world(ramesh, decmesh, 0)
    
    return xarray.DataArray(im.data, dims=dims, coords=coords, attrs=attrs)
