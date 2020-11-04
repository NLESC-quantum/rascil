""" Image operations visible to the Execution Framework as Components
"""

__all__ = ['image_gradients']

import warnings

from astropy.wcs import FITSFixedWarning

warnings.simplefilter('ignore', FITSFixedWarning)

from rascil.data_models.memory_data_models import Image

import logging
log = logging.getLogger('rascil-logger')

from rascil.processing_components.image.operations import create_empty_image_like


def image_gradients(im: Image):
    """Calculate image first order gradients numerically

    Two images are returned: one with respect to x and one with respect to y
    
    Gradient units are (incoming unit)/pixel e.g. Jy/beam/pixel
    
    :param im: Image
    :return: Gradient images
    """
    #assert isinstance(im, Image)

    nchan, npol, ny, nx = im["pixels"].data.shape
    
    gradientx = create_empty_image_like(im["pixels"].data)
    gradientx["pixels"].data[..., :, 1:nx] = im["pixels"].data[..., :, 1:nx] - im["pixels"].data[..., :, 0:(nx - 1)]
    gradienty = create_empty_image_like(im)
    gradienty["pixels"].data[..., 1:ny, :] = im["pixels"].data[..., 1:ny, :] - im["pixels"].data[..., 0:(ny - 1), :]
    
    return gradientx, gradienty
