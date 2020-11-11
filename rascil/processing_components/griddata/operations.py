#
"""
Functions that define and manipulate GridData.

The griddata has axes [chan, pol, z, y, x] where z, y, x are spatial axes in either sky or Fourier plane. The
order in the WCS is reversed so the grid_WCS describes UU, VV, WW, STOKES, FREQ axes.

GridData can be used to hold the Fourier transform of an Image or gridded visibilities. In addition, the convolution
function can be stored in a GridData, most probably with finer spatial sampling.


"""

__all__ = ['griddata_sizeof', 'create_griddata_from_image', 'create_griddata_from_array', 'copy_griddata',
           'convert_griddata_to_image', 'qa_griddata']

import copy
import logging

import xarray
import numpy
from astropy.wcs import WCS

from rascil.data_models.memory_data_models import GridData
from rascil.data_models.memory_data_models import QA
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.image.operations import create_image_from_array

log = logging.getLogger('rascil-logger')


def copy_griddata(gd):
    """ Copy griddata
    
    :param gd:
    :return:
    """
    #assert isinstance(gd, GridData), gd
    newgd = copy.deepcopy(gd)
    assert type(newgd) == GridData
    return newgd


def griddata_sizeof(gd: GridData):
    """ Return size in GB
    """
    return gd.data.nbytes() / 1024 / 1024 / 1024


def create_griddata_from_array(data: numpy.array, grid_wcs: WCS, projection_wcs: WCS,
                               polarisation_frame: PolarisationFrame) -> GridData:
    """ Create a griddata from an array and wcs's
    
    The griddata has axes [chan, pol, z, y, x] where z, y, x are spatial axes in either sky or Fourier plane. The
    order in the WCS is reversed so the grid_WCS describes UU, VV, WW, STOKES, FREQ axes
    
    Griddata holds the original sky plane projection in the projection_wcs.

    :param data: Numpy.array
    :param grid_wcs: Grid world coordinate system
    :param projection_wcs: Projection world coordinate system
    :param polarisation_frame: Polarisation Frame
    :return: GridData
    
    """

    log.debug("create_griddata_from_array: created %s image of shape %s" %
              (data.dtype, str(data.shape)))

    return GridData(data=data, grid_wcs=grid_wcs.deepcopy(), projection_wcs=projection_wcs.deepcopy(),
                    polarisation_frame=polarisation_frame)


def create_griddata_from_image(im, nw=1, wstep=1e15, polarisation_frame=None):
    """ Create a GridData from an image

    :param im: Template Image
    :param nw: Number of w planes
    :param wstep: Increment in w
    :return: GridData
    """
    
    if polarisation_frame is None:
        polarisation_frame = im.image_acc.polarisation_frame
    
    attrs = dict()
    attrs["projection_coords"] = im.coords
    
    nchan, npol, nv, nu = im["pixels"].data.shape
    pixels = numpy.zeros([nchan, npol, nw, nv, nu])
    
    cell_u = 1.0
    cell_v = 1.0
    cell_w = wstep
    
    coords = {
        "u": numpy.linspace(-nu // 2 * cell_u, +nu // 2 * cell_u, endpoint=False),
        "v": numpy.linspace(-nv // 2 * cell_v, +nv // 2 * cell_v, endpoint=False),
        "w": numpy.linspace(-nw // 2 * cell_w, +nw // 2 * cell_w, endpoint=False)
    }

    data_vars = dict()
    data_vars["pixels"] = xarray.DataArray(pixels, coords=coords, attrs=attrs)


def convert_griddata_to_image(gd):
    """ Convert griddata to an image
    
    :param gd:
    :return:
    """
    return create_image_from_array(gd.data, gd.grid_wcs, gd.polarisation_frame)


def qa_griddata(gd, context="") -> QA:
    """Assess the quality of a griddata

    :param gd:
    :return: QA
    """
    #assert isinstance(gd, GridData), gd
    data = {'shape': str(gd["pixels"].data.shape),
            'max': numpy.max(gd.data),
            'min': numpy.min(gd.data),
            'rms': numpy.std(gd.data),
            'sum': numpy.sum(gd.data),
            'medianabs': numpy.median(numpy.abs(gd.data)),
            'median': numpy.median(gd.data)}

    qa = QA(origin="qa_image", data=data, context=context)
    return qa
