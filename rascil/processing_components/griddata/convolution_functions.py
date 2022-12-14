#
"""
Functions that define and manipulate ConvolutionFunctions.

The griddata has axes [chan, pol, z, dy, dx, y, x] where z, y, x are spatial axes in either sky or Fourier plane. The
order in the WCS is reversed so the grid_WCS describes UU, VV, DUU, DVV, WW, STOKES, FREQ axes.

GridData can be used to hold the Fourier transform of an Image or gridded visibilities. In addition, the convolution
function can be stored in a GridData, most probably with finer spatial sampling.


"""

__all__ = [
    "create_convolutionfunction_from_image",
    "copy_convolutionfunction",
    "calculate_bounding_box_convolutionfunction",
    "apply_bounding_box_convolutionfunction",
    "qa_convolutionfunction",
    "export_convolutionfunction_to_fits",
]

import copy
import logging

import numpy
from astropy.io import fits
from astropy.wcs import WCS

from rascil.data_models.memory_data_models import ConvolutionFunction
from rascil.data_models.memory_data_models import QA

log = logging.getLogger("rascil-logger")


def create_convolutionfunction_from_image(
    im,
    nw=1,
    wstep=1e15,
    wtype="WW",
    oversampling=8,
    support=16,
    polarisation_frame=None,
):
    """Create a convolution function from an image

    The griddata has axes [chan, pol, z, dy, dx, y, x] where z, y, x are spatial axes in either sky or Fourier plane. The
    order in the WCS is reversed so the grid_WCS describes UU, VV, WW, STOKES, FREQ axes

    The axes UU,VV have the same physical stride as the image, The axes DUU, DVV are subsampled.

    Convolution function holds the original sky plane projection in the projection_wcs.

    :param im: Template Image
    :param nw: Number of z axes, usually z is W
    :param wstep: Step in z, usually z is W
    :param wtype: Type of z, usually 'WW'
    :param oversampling: Oversampling (size of dy, dx axes)
    :param support: Support of final convolution function (size of y, x axes)
    :return: Convolution Function

    """
    assert len(im["pixels"].data.shape) == 4

    assert im.image_acc.wcs.wcs.ctype[0] == "RA---SIN", im.image_acc.wcs.wcs.ctype[0]
    assert im.image_acc.wcs.wcs.ctype[1] == "DEC--SIN", im.image_acc.wcs.wcs.ctype[1]

    d2r = numpy.pi / 180.0

    # WCS Coords are [x, y, dy, dx, z, pol, chan] where x, y, z are spatial axes in real space or Fourier space
    # Array Coords are [chan, pol, z, dy, dx, y, x] where x, y, z are spatial axes in real space or Fourier space
    nchan, npol, ny, nx = im["pixels"].data.shape

    wcs = copy.deepcopy(im.image_acc.wcs.wcs)
    crval = wcs.crval
    crpix = wcs.crpix
    cdelt = wcs.cdelt
    ctype = wcs.ctype
    d2r = numpy.pi / 180.0
    cdelt[0] = 1.0 / (nx * cdelt[0] * d2r)
    cdelt[1] = 1.0 / (ny * cdelt[1] * d2r)

    cf_wcs = WCS(naxis=7)
    cf_wcs.wcs.crpix = [
        float(support // 2) + 1.0,
        float(support // 2) + 1.0,
        float(oversampling // 2) + 1.0,
        float(oversampling // 2) + 1.0,
        float(nw // 2 + 1.0),
        crpix[2],
        crpix[3],
    ]
    cf_wcs.wcs.ctype = ["UU", "VV", "DUU", "DVV", wtype, ctype[2], ctype[3]]
    cf_wcs.wcs.crval = [0.0, 0.0, 0.0, 0.0, 0.0, crval[2], crval[3]]
    cf_wcs.wcs.cdelt = [
        cdelt[0],
        cdelt[1],
        cdelt[0] / oversampling,
        cdelt[1] / oversampling,
        wstep,
        cdelt[2],
        cdelt[3],
    ]

    cf_wcs.wcs.radesys = "ICRS"
    cf_wcs.wcs.equinox = 2000.0

    cf_data = numpy.zeros(
        [nchan, npol, nw, oversampling, oversampling, support, support], dtype="complex"
    )

    if polarisation_frame is None:
        polarisation_frame = im.image_acc.polarisation_frame

    return ConvolutionFunction(
        data=cf_data, cf_wcs=cf_wcs, polarisation_frame=polarisation_frame
    )


def apply_bounding_box_convolutionfunction(cf, fractional_level=1e-4):
    """Apply a bounding box to a convolution function

    :param cf:
    :param fractional_level:
    :return: bounded convolution function
    """
    newcf = copy_convolutionfunction(cf)
    nx = newcf["pixels"].data.shape[-1]
    ny = newcf["pixels"].data.shape[-2]
    mask = numpy.max(numpy.abs(newcf["pixels"].data), axis=(0, 1, 2, 3, 4))
    mask /= numpy.max(mask)
    coords = numpy.argwhere(mask > fractional_level)
    crpx = int(numpy.round(cf.convolutionfunction_acc.cf_wcs.wcs.crpix[0]))
    crpy = int(numpy.round(cf.convolutionfunction_acc.cf_wcs.wcs.crpix[1]))
    x0, y0 = coords.min(axis=0, initial=cf["pixels"].data.shape[-1])
    dx = crpx - x0
    dy = crpy - y0
    x0 -= 1
    y0 -= 1
    x1 = crpx + dx - 1
    y1 = crpy + dy - 1
    nny, nnx = newcf["pixels"].data.shape[-2], newcf["pixels"].data.shape[-1]
    newcf.convolutionfunction_acc.cf_wcs.wcs.crpix[0] += nnx / 2 - nx / 2
    newcf.convolutionfunction_acc.cf_wcs.wcs.crpix[1] += nny / 2 - ny / 2

    newcf = ConvolutionFunction(
        data=newcf["pixels"].data[..., y0:y1, x0:x1],
        cf_wcs=newcf.convolutionfunction_acc.cf_wcs,
        polarisation_frame=newcf.convolutionfunction_acc.polarisation_frame,
    )
    return newcf


def calculate_bounding_box_convolutionfunction(cf, fractional_level=1e-4):
    """Calculate bounding boxes

    Returns a list of bounding boxes where each element is
    (z, (y0, y1), (x0, x1))

    These can be used in griddata/degridding.

    :param cf:
    :param fractional_level:
    :return: list of bounding boxes
    """
    bboxes = list()
    threshold = fractional_level * numpy.max(numpy.abs(cf["pixels"].data))
    for z in range(cf["pixels"].data.shape[2]):
        mask = numpy.max(numpy.abs(cf["pixels"].data[:, :, z, ...]), axis=(0, 1, 2, 3))
        coords = numpy.argwhere(mask > threshold)
        x0, y0 = coords.min(axis=0, initial=cf["pixels"].data.shape[-1])
        x1, y1 = coords.max(axis=0, initial=cf["pixels"].data.shape[-1])
        bboxes.append((z, (y0, y1), (x0, x1)))
    return bboxes


def qa_convolutionfunction(cf, context="") -> QA:
    """Assess the quality of a convolutionfunction

    :param cf:
    :return: QA
    """
    ##assert isinstance(cf, ConvolutionFunction), cf
    data = {
        "shape": str(cf["pixels"].data.shape),
        "max": numpy.max(cf.data),
        "min": numpy.min(cf.data),
        "rms": numpy.std(cf.data),
        "sum": numpy.sum(cf.data),
        "medianabs": numpy.median(numpy.abs(cf.data)),
        "median": numpy.median(cf.data),
    }

    qa = QA(origin="qa_image", data=data, context=context)
    return qa


def copy_convolutionfunction(cf):
    """Make a copy of a convolution function

    :param cf:
    :return:
    """
    ##assert isinstance(cf, ConvolutionFunction), cf
    return copy.deepcopy(cf)


def export_convolutionfunction_to_fits(
    cf: ConvolutionFunction, fitsfile: str = "cf.fits"
):
    """Write a convolution function to fits

    :param cf: Convolu
    :param fitsfile: Name of output fits file in storage
    :returns: None

    See also
        :py:func:`rascil.processing_components.image.operations.import_image_from_array`

    """
    ##assert isinstance(cf, ConvolutionFunction), cf
    return fits.writeto(
        filename=fitsfile,
        data=numpy.real(cf["pixels"].data),
        header=cf.convolutionfunction_acc.cf_wcs.to_header(),
        overwrite=True,
    )
