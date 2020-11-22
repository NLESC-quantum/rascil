#
"""
Functions that define and manipulate GridData.

The griddata has axes [chan, pol, z, y, x] where z, y, x are spatial axes in either sky or Fourier plane. The
order in the WCS is reversed so the grid_WCS describes UU, VV, WW, STOKES, FREQ axes.

GridData can be used to hold the Fourier transform of an Image or gridded visibilities. In addition, the convolution
function can be stored in a GridData, most probably with finer spatial sampling.


"""

__all__ = ['create_griddata_from_image', 'create_griddata_from_array', 'copy_griddata',
           'qa_griddata']

import copy
import logging

import numpy
from astropy.wcs import WCS

from rascil.data_models.memory_data_models import GridData
from rascil.data_models.memory_data_models import QA
from rascil.data_models.polarisation import PolarisationFrame

log = logging.getLogger('rascil-logger')


def copy_griddata(gd):
    """ Copy griddata
    
    :param gd:
    :return:
    """
    newgd = copy.deepcopy(gd)
    return newgd


def create_griddata_from_array(data: numpy.array, grid_wcs: WCS, polarisation_frame: PolarisationFrame) -> GridData:
    """ Create a griddata from an array and wcs's
    
    The griddata has axes [chan, pol, z, y, x] where z, y, x are spatial axes in either sky or Fourier plane. The
    order in the WCS is reversed so the grid_WCS describes UU, VV, WW, STOKES, FREQ axes
    
    Griddata holds the original sky plane projection in the projection_wcs.

    :param data: Numpy.array
    :param grid_wcs: Grid world coordinate system
    :param polarisation_frame: Polarisation Frame
    :return: GridData
    
    """
    
    log.debug("create_griddata_from_array: created %s image of shape %s" %
              (data.dtype, str(data.shape)))
    
    return GridData(data=data, polarisation_frame=polarisation_frame, grid_wcs=grid_wcs.deepcopy())


def create_griddata_from_image(im, polarisation_frame=None, ft_types=None):
    """ Create a GridData from an image

    :param im: Template Image
    :param nw: Number of w planes
    :param wstep: Increment in w
    :return: GridData
    """
    
    if ft_types is None:
        ft_types = ["UU", "VV"]
    nchan, npol, ny, nx = im["pixels"].shape
    gridshape = (nchan, npol, ny, nx)
    data = numpy.zeros(gridshape, dtype='complex')
    
    wcs = copy.deepcopy(im.image_acc.wcs)
    crval = wcs.wcs.crval
    crpix = wcs.wcs.crpix
    cdelt = wcs.wcs.cdelt
    ctype = wcs.wcs.ctype
    d2r = numpy.pi / 180.0
    cdelt[0] = 1.0 / (nx * cdelt[0] * d2r)
    cdelt[1] = 1.0 / (ny * cdelt[1] * d2r)
    
    # The negation in the longitude is needed by definition of RA, DEC
    grid_wcs = WCS(naxis=4)
    grid_wcs.wcs.crpix = [nx // 2 + 1, ny // 2 + 1, crpix[2], crpix[3]]
    grid_wcs.wcs.ctype = [ft_types[0], ft_types[1], ctype[2], ctype[3]]
    grid_wcs.wcs.crval = [0.0, 0.0, crval[2], crval[3]]
    grid_wcs.wcs.cdelt = [cdelt[0], cdelt[1], cdelt[2], cdelt[3]]
    grid_wcs.wcs.radesys = 'ICRS'
    grid_wcs.wcs.equinox = 2000.0
    
    if polarisation_frame is None:
        polarisation_frame = im.image_acc.polarisation_frame
    
    return GridData(data, polarisation_frame=polarisation_frame, grid_wcs=grid_wcs)


def qa_griddata(gd, context="") -> QA:
    """Assess the quality of a griddata

    :param gd:
    :return: QA
    """
    # assert isinstance(gd, GridData), gd
    data = {'shape': str(gd["pixels"].data.shape),
            'max': numpy.max(gd.data),
            'min': numpy.min(gd.data),
            'rms': numpy.std(gd.data),
            'sum': numpy.sum(gd.data),
            'medianabs': numpy.median(numpy.abs(gd.data)),
            'median': numpy.median(gd.data)}
    
    qa = QA(origin="qa_image", data=data, context=context)
    return qa
