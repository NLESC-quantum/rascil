""" Image operations visible to the Execution Framework as Components

"""

__all__ = ["export_xarray_to_fits", "import_xarray_from_fits"]

import logging
import warnings
from typing import List, Union

import numpy
import xarray
from astropy.io import fits
from astropy.wcs import FITSFixedWarning
from astropy.wcs import WCS

warnings.simplefilter("ignore", FITSFixedWarning)
log = logging.getLogger("rascil-logger")


def xarray_to_fits_header(xa: xarray.DataArray):
    """Convert xarray to FITS header

    :param xa:
    :return: fits_header
    """
    try:
        wcs = xa.attrs["wcs"]
    except:
        wcs = WCS(naxis=len(xa.shape))

        # This assumes that the coordinate system is linear!
        for icoord, coord in enumerate(xa.coords):
            fcoord = wcs.naxis - icoord - 1
            wcs.wcs.crpix[fcoord] = 0.0
            wcs.wcs.crval[fcoord] = xa.coords[coord].data[0]
            if xa.shape[icoord] > 1:
                wcs.wcs.cdelt[icoord] = (
                    xa.coords[coord].data[1] - xa.coords[coord].data[0]
                )
            else:
                wcs.wcs.cdelt[fcoord] = 0.0
            wcs.wcs.ctype[fcoord] = coord

    header = wcs.to_header()

    return header


def export_xarray_to_fits(
    xa: xarray.Dataset, fitsfile: Union[str, List] = "xarray.fits"
):
    """Write an image to fits

    :param xa: xarray.DataArray
    :param fitsfile: Name of output fits file in storage
    :returns: None

    See also
        :py:func:`rascil.processing_components.image.operations.import_image_from_fits`

    """
    ##assert isinstance(xa, xarray.DataArray), xa
    if xa["pixels"].data.dtype == "complex":
        assert len(fitsfile) == 2, "Need file names for real, imaginary parts"
        return fits.writeto(
            filename=fitsfile[0],
            data=numpy.real(xa["pixels"].data),
            header=xarray_to_fits_header(xa),
            overwrite=True,
        ) and fits.writeto(
            filename=fitsfile[1],
            data=numpy.imag(xa["pixels"].data),
            header=xarray_to_fits_header(xa),
            overwrite=True,
        )
    else:
        return fits.writeto(
            filename=fitsfile,
            data=xa["pixels"].data,
            header=xarray_to_fits_header(xa),
            overwrite=True,
        )


def import_xarray_from_fits(fitsfile: str) -> xarray.Dataset:
    """Read an xarray from fits

    :param fitsfile: FITS file in storage
    :return: xarray.DataArray

    See also
        :py:func:`rascil.processing_components.image.operations.import_image_from_fits`


    """
    warnings.simplefilter("ignore", FITSFixedWarning)
    hdulist = fits.open(fitsfile)
    data = hdulist[0].data
    wcs = WCS(fitsfile)
    hdulist.close()
    shape = list(data.shape)
    dims = wcs.axis_type_names[::-1]
    coords = {}
    for idim, dim in enumerate(dims):
        fdim = len(dims) - idim - 1
        coords[dim] = (
            wcs.wcs.crval[fdim]
            + (numpy.arange(shape[idim]) - wcs.wcs.crpix[fdim]) * wcs.wcs.cdelt[fdim]
        )

    attrs = {}
    attrs["wcs"] = wcs
    attrs["header"] = hdulist[0].header
    datavars = dict()
    datavars["pixels"] = xarray.DataArray(data, dims=dims, coords=coords)

    return xarray.Dataset(datavars, coords=coords, attrs=attrs)
