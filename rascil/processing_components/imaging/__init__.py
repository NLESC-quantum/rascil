""" Functions for imaging from visibility data.

The functions include 2D prediction and inversion operations. A very simple example, given a model Image to specify the image size, sampling, and phasecentre::

    model = create_image_from_visibility(vis, npixel=1024, nchan=1)
    dirty, sumwt = invert_blockvisibility(vis, model, context="2d")

The call to create_image_from_visibility step constructs a template image. The dirty image is constructed according to this template.

AW projection is supported by the predict_blockvisibility and invert_blockvisibility methods, provided the gridding
kernel is constructed and passed in as a partial. For example::

    gcfcf = functools.partial(create_awterm_convolutionfunction, nw=100, wstep=8.0, oversampling=8,
        support=100, use_aaf=True)
    dirty, sumwt = invert_blockvisibility(vis, model, context="awprojection", gcfcf=gcfcf)

If installed, the nifty gridder (https://gitlab.mpcdf.mpg.de/ift/nifty_gridder) can also be used::

    dirty, sumwt = invert_blockvisibility(vis, model, verbosity=2, context="ng")

The convolutional gridding functions are to be found in griddata module

    :py:mod:`rascil.processing_components.griddata`

"""
from .base import *
from .primary_beams import *
from .imaging_params import *
from .weighting import *
from .dft import *
from .ng import *
from .imaging import *
from .imaging_helpers import *
from .ao import *
from .wg import *

