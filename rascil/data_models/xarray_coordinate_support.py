""" Xarray coordinate support
"""

__all__ = ['']

from astropy.wcs import WCS

from rascil.data_models import PolarisationFrame

def image_wcs(ds):
    """
    
    :param ds:
    :return:
    """
    wcs = WCS()

    assert ds.rascil_data_model == "Image", ds.rascil_data_model
    
    w = WCS(naxis=4)
    nchan, npol, ny, nx = ds["pixels"].shape
    l = ds["x"].data[nx//2]
    m = ds["y"].data[ny//2]
    cellsize_l = (ds["x"].data[-1]-ds["x"].data[0])/(nx-1)
    cellsize_m = (ds["y"].data[-1]-ds["y"].data[0])/(ny-1)
    freq = ds["frequency"].data[0]
    pol = PolarisationFrame.fits_codes[ds.image_acc.polarisation_frame.type]
    if npol > 1:
        dpol = pol[1] - pol[0]
    else:
        dpol = 1.0
    if nchan > 1:
        channel_bandwidth = (ds["frequency"].data[-1]-ds["frequency"].data[0])/(nchan-1)
    else:
        channel_bandwidth = freq

    # The negation in the longitude is needed by definition of RA, DEC
    w.wcs.crpix = [nx // 2 + 1, ny // 2 + 1, 1.0, 1.0]
    w.wcs.ctype = ds.ctypes
    w.wcs.crval = [l, m, pol[0], freq]
    w.wcs.cdelt = [-cellsize_l, cellsize_m, dpol, channel_bandwidth]
    w.wcs.radesys = 'ICRS'
    w.wcs.equinox = 2000.0
    
    return w


def griddata_wcs(ds):
    """

    :param ds:
    :return:
    """
    assert ds.rascil_data_model == "GridData", ds.rascil_data_model
    
    # "frequency", "polarisation", "w", "v", "u"
    nchan, npol, nz, ny, nx = ds["pixels"].shape
    u = ds["u"].data[nx // 2]
    v = ds["v"].data[ny // 2]
    w = ds["w"].data[nz // 2]
    cellsize_u = (ds["u"].data[-1] - ds["u"].data[0]) / (nx - 1)
    cellsize_v = (ds["v"].data[-1] - ds["v"].data[0]) / (ny - 1)
    if nz > 1:
        cellsize_w = (ds["w"].data[-1] - ds["w"].data[0]) / (nz - 1)
    else:
        cellsize_w = 1e15
    freq = ds["frequency"].data[0]
    pol = PolarisationFrame.fits_codes[ds.image_acc.polarisation_frame.type]
    if npol > 1:
        dpol = pol[1] - pol[0]
    else:
        dpol = 1.0
    if nchan > 1:
        channel_bandwidth = (ds["frequency"].data[-1] - ds["frequency"].data[0]) / (nchan - 1)
    else:
        channel_bandwidth = freq
    
    # The negation in the longitude is needed by definition of RA, DEC
    wcs = WCS(naxis=5)
    wcs.wcs.crpix = [nx // 2 + 1, ny // 2 + 1, nz // 2 + 1.0, 1.0, 1.0]
    wcs.wcs.ctype = ["UU", "VV", 'WW', 'STOKES', 'FREQ']
    wcs.wcs.crval = [u, v, w, pol[0], freq]
    wcs.wcs.cdelt = [-cellsize_u, cellsize_v, cellsize_w, dpol, channel_bandwidth]
    wcs.wcs.radesys = 'ICRS'
    wcs.wcs.equinox = 2000.0
    
    return wcs


def cf_wcs(ds):
    """

    :param ds:
    :return:
    """
    assert ds.rascil_data_model == "ConvolutionFunction", ds.rascil_data_model
    
    # "frequency", "polarisation", "w", "v", "u"
    nchan, npol, ndv, ndu, nz, ny, nx = ds["pixels"].shape
    u = ds["u"].data[nx // 2]
    v = ds["v"].data[ny // 2]
    w = ds["w"].data[nz // 2]
    cellsize_u = (ds["u"].data[-1] - ds["u"].data[0]) / (nx - 1)
    cellsize_v = (ds["v"].data[-1] - ds["v"].data[0]) / (ny - 1)
    cellsize_du = (ds["du"].data[-1] - ds["du"].data[0]) / (ndu - 1)
    cellsize_dv = (ds["dv"].data[-1] - ds["dv"].data[0]) / (ndv - 1)
    if nz > 1:
        cellsize_w = (ds["w"].data[-1] - ds["w"].data[0]) / (nz - 1)
    else:
        cellsize_w = 1e15
    freq = ds["frequency"].data[0]
    pol = PolarisationFrame.fits_codes[ds.image_acc.polarisation_frame.type]
    if npol > 1:
        dpol = pol[1] - pol[0]
    else:
        dpol = 1.0
    if nchan > 1:
        channel_bandwidth = (ds["frequency"].data[-1] - ds["frequency"].data[0]) / (nchan - 1)
    else:
        channel_bandwidth = freq
    
    # The negation in the longitude is needed by definition of RA, DEC
    wcs = WCS(naxis=7)
    wcs.wcs.crpix = [nx // 2 + 1, ny // 2 + 1, ndu // 2 + 1, ndv // 2 + 1,
                     nz // 2 + 1.0, 1.0, 1.0]
    wcs.wcs.ctype = ["UU", "VV", "DUU", "DVV", "WW", "STOKES", "FREQ"]
    wcs.wcs.crval = [u, v, 0.0, 0.0, w, pol[0], freq]
    wcs.wcs.cdelt = [-cellsize_u, cellsize_v, cellsize_du, cellsize_dv, cellsize_w, dpol, channel_bandwidth]
    wcs.wcs.radesys = 'ICRS'
    wcs.wcs.equinox = 2000.0
    
    return wcs
