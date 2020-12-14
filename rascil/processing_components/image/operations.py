""" Image operations visible to the Execution Framework as Components

"""

__all__ = ['add_image',
           'calculate_image_frequency_moments',
           'calculate_image_from_frequency_moments',
           'convert_polimage_to_stokes',
           'convert_stokes_to_polimage',
           'create_empty_image_like',
           'create_image',
           'create_image_from_array',
           'create_w_term_like',
           'create_window',
           'export_image_to_fits',
           'fft_image_to_griddata',
           'image_is_canonical',
           'import_image_from_fits',
           'pad_image',
           'polarisation_frame_from_wcs',
           'qa_image',
           'remove_continuum_image',
           'reproject_image',
           'show_components',
           'show_image',
           'smooth_image',
           "scale_and_rotate_image",
           "apply_voltage_pattern_to_image"]

import copy
import logging
import warnings

import numpy
import xarray
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import FITSFixedWarning
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from reproject import reproject_interp

from rascil.data_models.memory_data_models import QA, Image
from rascil.data_models.parameters import get_parameter
from rascil.data_models.polarisation import PolarisationFrame, convert_stokes_to_linear, convert_stokes_to_circular, \
    convert_linear_to_stokes, convert_circular_to_stokes
from rascil.processing_components.calibration import apply_jones
from rascil.processing_components.fourier_transforms import w_beam, fft, ifft
from rascil.processing_components.griddata.operations import create_griddata_from_image

warnings.simplefilter('ignore', FITSFixedWarning)
log = logging.getLogger('rascil-logger')

def image_is_canonical(im: Image):
    """ Is this Image canonical format?

    :param im:
    :return:
    """
    if im is None:
        return True
    
    wcs = im.image_acc.wcs
    
    canonical = True
    canonical = canonical and len(im["pixels"].data.shape) == 4
    canonical = canonical and wcs.wcs.ctype[0] == 'RA---SIN' and wcs.wcs.ctype[1] == 'DEC--SIN'
    canonical = canonical and wcs.wcs.ctype[2] == 'STOKES'
    canonical = canonical and (wcs.wcs.ctype[3] == 'FREQ' or wcs.wcs.ctype[3] == "MOMENT")
    
    if not canonical:
        log.debug("image_is_canonical: Image is not canonical 4D image with axes RA---SIN, DEC--SIN, STOKES, FREQ")
        log.debug("image_is_canonical: axes are: {}".format(wcs.wcs.ctype))

    return canonical


def export_image_to_fits(im: Image, fitsfile: str = 'imaging.fits'):
    """ Write an image to fits

    :param im: Image
    :param fitsfile: Name of output fits file in storage
    :returns: None

    See also
        :py:func:`rascil.processing_components.image.operations.import_image_from_array`

    """
    ##assert isinstance(im, Image), im
    if im["pixels"].data.dtype == "complex":
        return fits.writeto(filename=fitsfile, data=numpy.real(im["pixels"].data),
                            header=im.image_acc.wcs.to_header(), overwrite=True)
    else:
        return fits.writeto(filename=fitsfile, data=im["pixels"].data,
                            header=im.image_acc.wcs.to_header(), overwrite=True)



def import_image_from_fits(fitsfile: str, fixpol=True) -> Image:
    """ Read an Image from fits

    :param fitsfile: FITS file in storage
    :return: Image

    See also
        :py:func:`rascil.processing_components.image.operations.export_image_to_array`


    """
    warnings.simplefilter('ignore', FITSFixedWarning)
    hdulist = fits.open(fitsfile)
    data = hdulist[0].data
    wcs = WCS(fitsfile)
    hdulist.close()
    
    polarisation_frame = PolarisationFrame('stokesI')
    frequency = numpy.array([1e8])
    
    if len(data.shape) == 4:
        try:
            polarisation_frame = polarisation_frame_from_wcs(wcs, data.shape)
            # FITS and RASCIL polarisation conventions differ
            if fixpol:
                permute = polarisation_frame.fits_to_rascil[polarisation_frame.type]
                newim_data = data.copy()
                for ip, p in enumerate(permute):
                    newim_data[:, p, ...] = data[:, ip, ...]
                data = newim_data
        
        except ValueError:
            polarisation_frame = PolarisationFrame('stokesI')

        try:
            w = wcs.sub(['spectral'])
            if len(data.shape) == 4:
                nchan = data.shape[0]
                frequency = w.wcs_pix2world(range(nchan), 0)[0]
            else:
                frequency = w.wcs_pix2world([0], 0)[0]

        except ValueError:
            frequency = numpy.array([1e8])
            
    elif len(data.shape) == 2:
        ny, nx = data.shape
        data.reshape([1, 1, ny, nx])

    try:
        phasecentre = SkyCoord(wcs.wcs.crval[0] * u.deg, wcs.wcs.crval[1] * u.deg)
    except ValueError:
        phasecentre = SkyCoord("0.0d", "0.0d")

    log.debug("import_image_from_fits: created %s image of shape %s" %
              (data.dtype, str(data.shape)))
    log.debug("import_image_from_fits: Max, min in %s = %.6f, %.6f" % (fitsfile, data.max(), data.min()))

    return Image(data=data, polarisation_frame=polarisation_frame, wcs=wcs)


def reproject_image(im: Image, newwcs: WCS, shape=None) -> (Image, Image):
    """ Re-project an image to a new coordinate system

    Currently uses the reproject python package. This seems to have some features do be careful using this method.
    For timeslice imaging griddata is used.

    :param im: Image to be reprojected
    :param newwcs: New WCS
    :param shape: Desired shape
    :return: Reprojected Image, Footprint Image
    """
    
    ##assert isinstance(im, Image), im
    
    if len(im["pixels"].shape) == 4:
        nchan, npol, ny, nx = im["pixels"].shape
        if im["pixels"].data.dtype == 'complex':
            rep_real = numpy.zeros(shape, dtype='float')
            rep_imag = numpy.zeros(shape, dtype='float')
            foot = numpy.zeros(shape, dtype='float')
            for chan in range(nchan):
                for pol in range(npol):
                    rep_real[chan, pol], foot[chan, pol] = reproject_interp((im["pixels"].data.real[chan, pol],
                                                                             im.image_acc.wcs.sub(2)),
                                                                            newwcs.sub(2), shape[2:], order='bicubic')
                    rep_imag[chan, pol], foot[chan, pol] = reproject_interp((im["pixels"].data.imag[chan, pol],
                                                                             im.image_acc.wcs.sub(2)),
                                                                            newwcs.sub(2), shape[2:], order='bicubic')
            rep = rep_real + 1j * rep_imag
        else:
            rep = numpy.zeros(shape, dtype='float')
            foot = numpy.zeros(shape, dtype='float')
            for chan in range(nchan):
                for pol in range(npol):
                    rep[chan, pol], foot[chan, pol] = reproject_interp((im["pixels"].data[chan, pol],
                                                                        im.image_acc.wcs.sub(2)),
                                                                       newwcs.sub(2), shape[2:], order='bicubic')
        
        if numpy.sum(foot.data) < 1e-12:
            log.warning("reproject_image: no valid points in reprojection")
    elif len(im["pixels"].data.shape) == 2:
        if im["pixels"].data.dtype == 'complex':
            rep_real, foot = reproject_interp((im["pixels"].data.real, im.image_acc.wcs), newwcs, shape, order='bicubic')
            rep_imag, foot = reproject_interp((im["pixels"].data.imag, im.image_acc.wcs), newwcs, shape, order='bicubic')
            rep = rep_real + 1j * rep_imag
        else:
            rep, foot = reproject_interp((im["pixels"].data, im.image_acc.wcs), newwcs, shape, order='bicubic')
        
        if numpy.sum(foot.data) < 1e-12:
            log.warning("reproject_image: no valid points in reprojection")
    
    else:
        raise ValueError("Cannot reproject image with shape {}".format(im["pixels"].shape))
    rep = numpy.nan_to_num(rep)
    foot = numpy.nan_to_num(foot)
    return create_image_from_array(rep,  newwcs, im.image_acc.polarisation_frame), \
           create_image_from_array(foot, newwcs, im.image_acc.polarisation_frame)


def add_image(im1: Image, im2: Image) -> Image:
    """ Add two images

    :param im1: Image
    :param im2: Image
    :return: Image
    """
    return create_image_from_array(im1["pixels"].data+ im2["pixels"].data,
                                   im1.image_acc.wcs,
                                   im1.image_acc.polarisation_frame)


def qa_image(im: Image, context="") -> QA:
    """Assess the quality of an image

    QA is a standard set of statistics of an image; max, min, maxabs, rms, sum, medianabs, medianabsdevmedian, median

    :param im:
    :return: QA
    """
    ##assert isinstance(im, Image), im
    im_data = im["pixels"].data
    data = {'shape': str(im["pixels"].data.shape),
            'max': numpy.max(im_data),
            'min': numpy.min(im_data),
            'maxabs': numpy.max(numpy.abs(im_data)),
            'rms': numpy.std(im_data),
            'sum': numpy.sum(im_data),
            'medianabs': numpy.median(numpy.abs(im_data)),
            'medianabsdevmedian': numpy.median(numpy.abs(im_data - numpy.median(im_data))),
            'median': numpy.median(im_data)}
    
    qa = QA(origin="qa_image", data=data, context=context)
    return qa


def show_image(im: Image, fig=None, title: str = '', pol=0, chan=0, cm='Greys', components=None,
                vmin=None, vmax=None, vscale=1.0):
    """ Show an Image with coordinates using matplotlib, optionally with components

    :param im: Image
    :param fig: Matplotlib figure
    :param title: String for title of plot
    :param pol: Polarisation to show (index)
    :param chan: Channel to show (index)
    :param components: Optional components to be overlaid
    :param vmin: Clip to this minimum
    :param vmax: Clip to this maximum
    :param vscale: scale max, min by this amount
    :return:
    """
    import matplotlib.pyplot as plt
    
    ##assert isinstance(im, Image), im
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=im.image_acc.wcs.sub([1, 2]))
    
    if len(im["pixels"].data.shape) == 4:
        data_array = numpy.real(im["pixels"].data[chan, pol, :, :])
    else:
        data_array = numpy.real(im["pixels"].data)
    
    if vmax is None:
        vmax = vscale * numpy.max(data_array)
    if vmin is None:
        vmin = vscale * numpy.min(data_array)
    
    cm = ax.imshow(data_array, origin='lower', cmap=cm, vmax=vmax, vmin=vmin)
    
    ax.set_xlabel(im.image_acc.wcs.wcs.ctype[0])
    ax.set_ylabel(im.image_acc.wcs.wcs.ctype[1])
    ax.set_title(title)
    
    fig.colorbar(cm, orientation='vertical', shrink=0.7)
    
    if components is not None:
        for sc in components:
            x, y = skycoord_to_pixel(sc.direction, im.image_acc.wcs, 0, 'wcs')
            ax.plot(x, y, marker='+', color='red')
    
    return fig


def show_components(im, comps, npixels=128, fig=None, vmax=None, vmin=None, title=''):
    """ Show components against an image

    :param im:
    :param comps:
    :param npixels:
    :param fig:
    :return:
    """
    import matplotlib.pyplot as plt
    
    if vmax is None:
        vmax = numpy.max(im["pixels"].data[0, 0, ...])
    if vmin is None:
        vmin = numpy.min(im["pixels"].data[0, 0, ...])
    
    if not fig:
        fig = plt.figure()
    plt.clf()
    
    assert image_is_canonical(im)
    
    for isc, sc in enumerate(comps):
        newim = im.copy(deep=True)
        plt.subplot(111, projection=newim.image_acc.wcs.sub([1, 2]))
        centre = numpy.round(skycoord_to_pixel(sc.direction, newim.image_acc.wcs, 1, 'wcs')).astype('int')
        newim["pixels"].data = \
            newim["pixels"].data[:, :, (centre[1] - npixels // 2):(centre[1] + npixels // 2),
            (centre[0] - npixels // 2):(centre[0] + npixels // 2)]
        newim.image_acc.wcs.wcs.crpix[0] -= centre[0] - npixels // 2
        newim.image_acc.wcs.wcs.crpix[1] -= centre[1] - npixels // 2
        plt.imshow(newim["pixels"].data[0, 0, ...], origin='lower', cmap='Greys', vmax=vmax, vmin=vmin)
        x, y = skycoord_to_pixel(sc.direction, newim.image_acc.wcs, 0, 'wcs')
        plt.plot(x, y, marker='+', color='red')
        plt.title('Name = %s, flux = %s' % (sc.name, sc.flux))
        plt.show()


def smooth_image(model: Image, width=1.0, normalise=True):
    """ Smooth an image with a 2D Gaussian kernel

    :param model: Image
    :param width: Kernel width in pixels
    :param normalise: Normalise kernel peak to unity

    """
    ##assert isinstance(model, Image), model
    assert image_is_canonical(model)
    
    from astropy.convolution.kernels import Gaussian2DKernel
    from astropy.convolution import convolve_fft
    
    kernel = Gaussian2DKernel(width)
    model_type = model["pixels"].data.dtype
    
    cmodel = create_empty_image_like(model)
    nchan, npol, _, _ = model.image_acc.shape
    for pol in range(npol):
        for chan in range(nchan):
            cmodel["pixels"].data[chan, pol, :, :] = convolve_fft(model["pixels"].data[chan, pol, :, :], kernel,
                                                        normalize_kernel=False,
                                                        allow_huge=True)
    # The convolve_fft step seems to return an object dtype
    cmodel["pixels"].data = cmodel["pixels"].data.astype(model_type)
    if normalise and isinstance(kernel, Gaussian2DKernel):
        cmodel["pixels"].data *= 2 * numpy.pi * width ** 2
    
    return cmodel


def calculate_image_frequency_moments(im: Image, reference_frequency=None, nmoment=1) -> Image:
    """Calculate frequency weighted moments of an image cube

    The frequency moments are calculated using:

    .. math::

        w_k = \\left(\\left(\\nu - \\nu_{ref}\\right) /  \\nu_{ref}\\right)^k


    Note that the spectral axis is replaced by a MOMENT axis.

    For example, to find the moments and then reconstruct from just the moments::

        moment_cube = calculate_image_frequency_moments(model_multichannel, nmoment=5)
        reconstructed_cube = calculate_image_from_frequency_moments(model_multichannel, moment_cube)

    :param im: Image cube
    :param reference_frequency: Reference frequency (default None uses average)
    :param nmoment: Number of moments to calculate
    :return: Moments image
    """
    #assert isinstance(im, Image)
    assert image_is_canonical(im)
    
    assert nmoment > 0
    nchan, npol, ny, nx = im["pixels"].data.shape
    channels = numpy.arange(nchan)
    freq = im.image_acc.wcs.sub(['spectral']).wcs_pix2world(channels, 0)[0]
    
    assert nmoment <= nchan, "Number of moments %d cannot exceed the number of channels %d" % (nmoment, nchan)
    
    if reference_frequency is None:
        reference_frequency = numpy.average(freq.data)
    log.debug("calculate_image_frequency_moments: Reference frequency = %.3f (MHz)" % (reference_frequency / 1e6))
    
    moment_data = numpy.zeros([nmoment, npol, ny, nx])

    assert not numpy.isnan(numpy.sum(im["pixels"].data)), "NaNs present in image data"

    for moment in range(nmoment):
        for chan in range(nchan):
            weight = numpy.power((freq[chan] - reference_frequency) / reference_frequency, moment)
            moment_data[moment, ...] += im["pixels"].data[chan, ...] * weight
    
    assert not numpy.isnan(numpy.sum(moment_data)), "NaNs present in moment data"

    moment_wcs = copy.deepcopy(im.image_acc.wcs)
    
    moment_wcs.wcs.ctype[3] = 'MOMENT'
    moment_wcs.wcs.crval[3] = 0.0
    moment_wcs.wcs.crpix[3] = 1.0
    moment_wcs.wcs.cdelt[3] = 1.0
    moment_wcs.wcs.cunit[3] = ''
    
    return create_image_from_array(moment_data, moment_wcs, im.image_acc.polarisation_frame)


def calculate_image_from_frequency_moments(im: Image, moment_image: Image, reference_frequency=None) -> Image:
    """Calculate channel image from frequency weighted moments

    .. math::

        w_k = \\left(\\left(\\nu - \\nu_{ref}\\right) /  \\nu_{ref}\\right)^k


    Note that a new image is created

    For example, to find the moments and then reconstruct from just the moments::

        moment_cube = calculate_image_frequency_moments(model_multichannel, nmoment=5)
        reconstructed_cube = calculate_image_from_frequency_moments(model_multichannel, moment_cube)


    :param im: Image cube to be reconstructed
    :param moment_image: Moment cube (constructed using calculate_image_frequency_moments)
    :param reference_frequency: Reference frequency (default None uses average)
    :return: reconstructed image
    """
    #assert isinstance(im, Image)
    nchan, npol, ny, nx = im["pixels"].data.shape
    nmoment, mnpol, mny, mnx = moment_image["pixels"].data.shape
    assert nmoment > 0
    
    assert npol == mnpol
    assert ny == mny
    assert nx == mnx
    
    #assert moment_image.wcs.wcs.ctype[3] == 'MOMENT', "Second image should be a moment image"
    
    
    if reference_frequency is None:
        reference_frequency = numpy.average(im.frequency.data)
    log.debug("calculate_image_from_frequency_moments: Reference frequency = %.3f (MHz)" % (1e-6 * reference_frequency))
    
    newim_wcs = im.image_acc.wcs
    newim_wcs.wcs.ctype[3] = "MOMENT"
    newim_wcs.wcs.crval[3] = 1
    newim_wcs.wcs.crpix[3] = 1
    newim_wcs.wcs.cdelt[3] = 1

    newim_data = numpy.zeros_like(im["pixels"].data[...])
    for moment in range(nmoment):
        for chan in range(nchan):
            weight = numpy.power((im.frequency[chan].data - reference_frequency) / reference_frequency, moment)
            newim_data[chan, ...] += moment_image["pixels"].data[moment, ...] * weight
    
    newim = create_image_from_array(newim_data, wcs=newim_wcs,
                                    polarisation_frame=im.image_acc.polarisation_frame)
    return newim


def remove_continuum_image(im: Image, degree=1, mask=None):
    """ Fit and remove continuum visibility in place

    Fit a polynomial in frequency of the specified degree where mask is True and remove it from the image

    :param im:
    :param degree: 1 is a constant, 2 is a slope, etc.
    :param mask: Frequency mask
    :return:
    """
    #assert isinstance(im, Image)
    assert image_is_canonical(im)
    
    if mask is not None:
        assert numpy.sum(mask) > 2 * degree, "Insufficient channels for fit"
    
    nchan, npol, ny, nx = im["pixels"].data.shape
    channels = numpy.arange(nchan)
    frequency = im.image_acc.wcs.sub(['spectral']).wcs_pix2world(channels, 0)[0]
    frequency -= frequency[nchan // 2]
    frequency /= numpy.max(frequency)
    wt = numpy.ones_like(frequency)
    if mask is not None:
        wt[mask] = 0.0
    
    for pol in range(npol):
        for y in range(ny):
            for x in range(nx):
                fit = numpy.polyfit(frequency, im["pixels"].data[:, pol, y, x], w=wt, deg=degree)
                prediction = numpy.polyval(fit, frequency)
                im["pixels"].data[:, pol, y, x] -= prediction
    return im


def convert_stokes_to_polimage(im: Image, polarisation_frame: PolarisationFrame):
    """Convert a stokes image in IQUV to polarisation_frame

    For example::
        impol = convert_stokes_to_polimage(imIQUV, Polarisation_Frame('linear'))

    :param im: Image to be converted
    :param polarisation_frame: desired polarisation frame
    :returns: Complex image

    See also
        :py:func:`rascil.processing_components.image.operations.convert_polimage_to_stokes`
        :py:func:`rascil.data_models.polarisation.convert_circular_to_stokes`
        :py:func:`rascil.data_models.polarisation.convert_linear_to_stokes`
    """
    
    if polarisation_frame == PolarisationFrame('linear'):
        cimarr = convert_stokes_to_linear(im["pixels"].data)
        return create_image_from_array(cimarr, im.image_acc.wcs, polarisation_frame)
    elif polarisation_frame == PolarisationFrame('linearnp'):
        cimarr = convert_stokes_to_linear(im["pixels"].data)
        return create_image_from_array(cimarr, im.image_acc.wcs, polarisation_frame)
    elif polarisation_frame == PolarisationFrame('circular'):
        cimarr = convert_stokes_to_circular(im["pixels"].data)
        return create_image_from_array(cimarr, im.image_acc.wcs, polarisation_frame)
    elif polarisation_frame == PolarisationFrame('circularnp'):
        cimarr = convert_stokes_to_circular(im["pixels"].data)
        return create_image_from_array(cimarr, im.image_acc.wcs, polarisation_frame)
    elif polarisation_frame == PolarisationFrame('stokesI'):
        return create_image_from_array(im["pixels"].data.astype("complex"), im.image_acc.wcs, PolarisationFrame('stokesI'))
    else:
        raise ValueError("Cannot convert stokes to %s" % (polarisation_frame.type))


def convert_polimage_to_stokes(im: Image, complex_image=False, **kwargs):
    """Convert a polarisation image to stokes IQUV (complex)

    For example:
        imIQUV = convert_polimage_to_stokes(impol)

    :param im: Complex Image in linear or circular
    :param complex_image: Return complex image?
    :returns: Complex or Real image

    See also
        :py:func:`rascil.processing_components.image.operations.convert_stokes_to_polimage`
        :py:func:`rascil.data_models.polarisation.convert_stokes_to_circular`
        :py:func:`rascil.data_models.polarisation.convert_stokes_to_linear`

    """
    #assert isinstance(im, Image)
    assert im["pixels"].data.dtype == 'complex', im["pixels"].data.dtype
    
    def to_required(cimarr):
        if complex_image:
            return cimarr
        else:
            return numpy.real(cimarr)
    
    if im.image_acc.polarisation_frame == PolarisationFrame('linear'):
        cimarr = convert_linear_to_stokes(im["pixels"].data)
        return create_image_from_array(to_required(cimarr), im.image_acc.wcs, PolarisationFrame('stokesIQUV'))
    elif im.image_acc.polarisation_frame == PolarisationFrame('linearnp'):
        cimarr = convert_linear_to_stokes(im["pixels"].data)
        return create_image_from_array(to_required(cimarr), im.image_acc.wcs, PolarisationFrame('stokesIQ'))
    elif im.image_acc.polarisation_frame == PolarisationFrame('circular'):
        cimarr = convert_circular_to_stokes(im["pixels"].data)
        return create_image_from_array(to_required(cimarr), im.image_acc.wcs, PolarisationFrame('stokesIQUV'))
    elif im.image_acc.polarisation_frame == PolarisationFrame('circularnp'):
        cimarr = convert_circular_to_stokes(im["pixels"].data)
        return create_image_from_array(to_required(cimarr), im.image_acc.wcs, PolarisationFrame('stokesIV'))
    elif im.image_acc.polarisation_frame == PolarisationFrame('stokesI'):
        return create_image_from_array(to_required(im["pixels"].data), im.image_acc.wcs, PolarisationFrame('stokesI'))
    else:
        raise ValueError("Cannot convert %s to stokes" % (im.image_acc.polarisation_frame.type))


def create_window(template, window_type, **kwargs):
    """Create a window image using one of a number of methods

    The window is 1.0 or 0.0

    window types:
        'quarter': Inner quarter of the image

        'no_edge': 'window_edge' pixels around edge set to zero

        'threshold': template image pixels < 'window_threshold' absolute value set to zero

    :param template: Template image
    :param window_type: 'quarter' | 'no_edge' | 'threshold'
    :return: New image containing window

    See also
        :py:func:`rascil.processing_components.image.deconvolution.deconvolve_cube`


    """
    
    assert image_is_canonical(template)
    
    window = create_empty_image_like(template)
    if window_type == 'quarter':
        qx = template["pixels"].shape[3] // 4
        qy = template["pixels"].shape[2] // 4
        window["pixels"].data[..., (qy + 1):3 * qy, (qx + 1):3 * qx] = 1.0
        log.info('create_mask: Cleaning inner quarter of each sky plane')
    elif window_type == 'no_edge':
        edge = get_parameter(kwargs, 'window_edge', 16)
        nx = template["pixels"].shape[3]
        ny = template["pixels"].shape[2]
        window["pixels"].data[..., (edge + 1):(ny - edge), (edge + 1):(nx - edge)] = 1.0
        log.info('create_mask: Window omits %d-pixel edge of each sky plane' % (edge))
    elif window_type == 'threshold':
        window_threshold = get_parameter(kwargs, 'window_threshold', None)
        if window_threshold is None:
            window_threshold = 10.0 * numpy.std(template["pixels"].data)
        window["pixels"].data[template["pixels"].data >= window_threshold] = 1.0
        log.info('create_mask: Window omits all points below %g' % (window_threshold))
    elif window_type is None:
        log.info("create_mask: Mask covers entire image")
    else:
        raise ValueError("Window shape %s is not recognized" % window_type)
    
    return window

def create_image(npixel=512,
                 cellsize=0.000015,
                 polarisation_frame=PolarisationFrame("stokesI"),
                 frequency=numpy.array([1e8]),
                 channel_bandwidth=numpy.array([1e6]),
                 phasecentre=None,
                 nchan=None,
                 dtype='float64') -> Image:
    """Create an empty  image consistent with the inputs.

    :param npixel: Number of pixels
    :param cellsize: cellsize in radians
    :param polarisation_frame: Polarisation frame (default PolarisationFrame("stokesI"))
    :param frequency: Array of frequencies (Hz)
    :param channel_bandwidth: Array of Channel width (Hz)
    :param phasecentre: phasecentre (SkyCoord)
    :param nchan: Number of channels in image
    :param dtype: Python data type for array
    :return: Image

    See also
        :py:func:`rascil.processing_components.image.operations.testing_support.create_image_from_array`
        :py:func:`rascil.processing_components.imaging.base.create_image_from_visibility`
        :py:func:`rascil.processing_components.simulation.create_test_image`
        :py:mod:`rascil.processing_components.simulation`

    """
    
    if phasecentre is None:
        raise ValueError("phasecentre must be specified")
    
    if polarisation_frame is None:
        polarisation_frame = PolarisationFrame("stokesI")
    
    npol = polarisation_frame.npol
    if nchan is None:
        nchan = len(frequency)
    
    shape = [nchan, npol, npixel, npixel]
    w = WCS(naxis=4)
    pol = PolarisationFrame.fits_codes[polarisation_frame.type]
    if npol > 1:
        dpol = pol[1] - pol[0]
    else:
        dpol = 1.0
        
    # The negation in the longitude is needed by definition of RA, DEC
    w.wcs.cdelt = [-cellsize * 180.0 / numpy.pi, cellsize * 180.0 / numpy.pi, dpol, channel_bandwidth[0]]
    w.wcs.crpix = [npixel // 2 + 1, npixel // 2 + 1, pol[0], 1.0]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", 'STOKES', 'FREQ']
    w.wcs.crval = [phasecentre.ra.deg, phasecentre.dec.deg, 1.0, frequency[0]]
    w.naxis = 4
    w.wcs.radesys = 'ICRS'
    w.wcs.equinox = 2000.0
    
    return Image(numpy.zeros(shape, dtype=dtype), wcs=w, polarisation_frame=polarisation_frame)


def create_image_from_array(data: numpy.array, wcs: WCS, polarisation_frame: PolarisationFrame,
                            chunksize=None) -> Image:
    """ Create an image from an array and optional wcs

    The output image preserves a reference to the input array.

    :param data: Numpy.array
    :param wcs: World coordinate system
    :param polarisation_frame: Polarisation Frame
    :param chunksize: Size of xarray chunking
    :return: Image

    See also
        :py:func:`rascil.processing_components.image.operations.create_image`
        :py:func:`rascil.processing_components.imaging.base.create_image_from_visibility`

    """

    return Image(data=data, polarisation_frame=polarisation_frame, wcs=wcs)

def polarisation_frame_from_wcs(wcs, shape) -> PolarisationFrame:
    """Convert wcs to polarisation_frame

    See FITS definition in Table 29 of https://fits.gsfc.nasa.gov/standard40/fits_standard40draft1.pdf
    or subsequent revision

        1 I Standard Stokes unpolarized
        2 Q Standard Stokes linear
        3 U Standard Stokes linear
        4 V Standard Stokes circular
        −1 RR Right-right circular
        −2 LL Left-left circular
        −3 RL Right-left cross-circular
        −4 LR Left-right cross-circular
        −5 XX X parallel linear
        −6 YY Y parallel linear
        −7 XY XY cross linear
        −8 YX YX cross linear

        stokesI [1]
        stokesIQUV [1,2,3,4]
        circular [-1,-2,-3,-4]
        linear [-5,-6,-7,-8]

    For example::
        pol_frame = polarisation_frame_from_wcs(im.image_acc.wcs, im["pixels"].data.shape)


    :param wcs: World Coordinate System
    :param shape: Shape corresponding to wcs
    :returns: Polarisation_Frame object
    """
    # The third axis should be stokes:
    
    polarisation_frame = None
    
    if len(shape) == 2:
        polarisation_frame = PolarisationFrame("stokesI")
    else:
        npol = shape[1]
        pol = wcs.sub(['stokes']).wcs_pix2world(range(npol), 0)[0]
        pol = numpy.array(pol, dtype='int')
        for key in PolarisationFrame.fits_codes.keys():
            keypol = numpy.array(PolarisationFrame.fits_codes[key])
            if numpy.array_equal(pol, keypol):
                polarisation_frame = PolarisationFrame(key)
                return polarisation_frame
    if polarisation_frame is None:
        raise ValueError("Cannot determine polarisation code")
    
    #assert isinstance(polarisation_frame, PolarisationFrame)
    return polarisation_frame


def create_empty_image_like(im: Image) -> Image:
    """ Create an empty image like another in shape and wcs

    The data array is initialized to zero

    :param im:
    :return: Image

    """
    return create_image_from_array(numpy.zeros_like(im["pixels"].data),
                                   wcs=im.image_acc.wcs,
                                   polarisation_frame=im.image_acc.polarisation_frame)


def fft_image_to_griddata(im):
    """ WCS-aware FFT of a canonical image

    The only transforms supported are:
        RA--SIN, DEC--SIN <-> UU, VV
        XX, YY <-> KX, KY

    For example::

        from rascil.processing_components import create_test_image, fft_image_to_griddata
        im = create_test_image()
        print(im)
            Image:
                Shape: (1, 1, 256, 256)
                WCS: WCS Keywords
            Number of WCS axes: 4
            CTYPE : 'RA---SIN'  'DEC--SIN'  'STOKES'  'FREQ'
            CRVAL : 0.0  35.0  1.0  100000000.0
            CRPIX : 129.0  129.0  1.0  1.0
            PC1_1 PC1_2 PC1_3 PC1_4  : 1.0  0.0  0.0  0.0
            PC2_1 PC2_2 PC2_3 PC2_4  : 0.0  1.0  0.0  0.0
            PC3_1 PC3_2 PC3_3 PC3_4  : 0.0  0.0  1.0  0.0
            PC4_1 PC4_2 PC4_3 PC4_4  : 0.0  0.0  0.0  1.0
            CDELT : -0.000277777791  0.000277777791  1.0  100000.0
            NAXIS : 0  0
                Polarisation frame: stokesI
        print(fft_image_to_griddata(im))
            Image:
                Shape: (1, 1, 256, 256)
                WCS: WCS Keywords
            Number of WCS axes: 4
            CTYPE : 'UU'  'VV'  'STOKES'  'FREQ'
            CRVAL : 0.0  0.0  1.0  100000000.0
            CRPIX : 129.0  129.0  1.0  1.0
            PC1_1 PC1_2 PC1_3 PC1_4  : 1.0  0.0  0.0  0.0
            PC2_1 PC2_2 PC2_3 PC2_4  : 0.0  1.0  0.0  0.0
            PC3_1 PC3_2 PC3_3 PC3_4  : 0.0  0.0  1.0  0.0
            PC4_1 PC4_2 PC4_3 PC4_4  : 0.0  0.0  0.0  1.0
            CDELT : -805.7218610503596  805.7218610503596  1.0  100000.0
            NAXIS : 0  0
                Polarisation frame: stokesI

    :param im:
    :return:

    See also
        :py:func:`rascil.processing_components.fourier_transforms.fft_support.fft`
        :py:func:`rascil.processing_components.fourier_transforms.fft_support.ifft`
    """
    assert im.attrs["rascil_data_model"] == "Image"
    
    assert len(im["pixels"].data.shape) == 4
    wcs = im.image_acc.wcs
    
    if wcs.wcs.ctype[0] == 'RA---SIN' and wcs.wcs.ctype[1] == 'DEC--SIN':
        ft_types = ['UU', 'VV']
    elif wcs.wcs.ctype[0] == 'XX' and wcs.wcs.ctype[1] == 'YY':
        ft_types = ['KX', 'KY']
    elif wcs.wcs.ctype[0] == 'AZELGEO long' and wcs.wcs.ctype[1] == 'AZELGEO lati':
        ft_types = ['KX', 'KY']
    else:
        raise NotImplementedError("Cannot FFT specified axes {0}, {1}".format(wcs.wcs.ctype[0], wcs.wcs.ctype[1]))

    gd = create_griddata_from_image(im, ft_types=ft_types)
    gd["pixels"].data = ifft(im["pixels"].data.astype('complex'))
    return gd

def ifft_griddata_to_image(gd, template):
    """ WCS-aware FFT of a canonical image

    The only transforms supported are:
        RA--SIN, DEC--SIN <-> UU, VV
        XX, YY <-> KX, KY

    For example::

        from rascil.processing_components import create_test_image, fft_image_to_griddata
        im = create_test_image()
        print(im)
            Image:
                Shape: (1, 1, 256, 256)
                WCS: WCS Keywords
            Number of WCS axes: 4
            CTYPE : 'RA---SIN'  'DEC--SIN'  'STOKES'  'FREQ'
            CRVAL : 0.0  35.0  1.0  100000000.0
            CRPIX : 129.0  129.0  1.0  1.0
            PC1_1 PC1_2 PC1_3 PC1_4  : 1.0  0.0  0.0  0.0
            PC2_1 PC2_2 PC2_3 PC2_4  : 0.0  1.0  0.0  0.0
            PC3_1 PC3_2 PC3_3 PC3_4  : 0.0  0.0  1.0  0.0
            PC4_1 PC4_2 PC4_3 PC4_4  : 0.0  0.0  0.0  1.0
            CDELT : -0.000277777791  0.000277777791  1.0  100000.0
            NAXIS : 0  0
                Polarisation frame: stokesI
        print(fft_image_to_griddata(im))
            Image:
                Shape: (1, 1, 256, 256)
                WCS: WCS Keywords
            Number of WCS axes: 4
            CTYPE : 'UU'  'VV'  'STOKES'  'FREQ'
            CRVAL : 0.0  0.0  1.0  100000000.0
            CRPIX : 129.0  129.0  1.0  1.0
            PC1_1 PC1_2 PC1_3 PC1_4  : 1.0  0.0  0.0  0.0
            PC2_1 PC2_2 PC2_3 PC2_4  : 0.0  1.0  0.0  0.0
            PC3_1 PC3_2 PC3_3 PC3_4  : 0.0  0.0  1.0  0.0
            PC4_1 PC4_2 PC4_3 PC4_4  : 0.0  0.0  0.0  1.0
            CDELT : -805.7218610503596  805.7218610503596  1.0  100000.0
            NAXIS : 0  0
                Polarisation frame: stokesI

    :param gd: Input GridData
    :param template_image: Template output image
    :return: Image

    See also
        :py:func:`rascil.processing_components.fourier_transforms.fft_support.fft`
        :py:func:`rascil.processing_components.fourier_transforms.fft_support.ifft`
    """
    assert len(gd["pixels"].data.shape) == 4
    wcs = gd.griddata_acc.griddata_wcs
    template_wcs = template.image_acc.wcs
    ft_wcs = copy.deepcopy(template_wcs)
    
    if wcs.wcs.ctype[0] == 'UU' and wcs.wcs.ctype[1] == 'VV':
        ft_wcs.wcs.ctype[0] = template_wcs.wcs.ctype[0]
        ft_wcs.wcs.ctype[1] = template_wcs.wcs.ctype[1]
    elif wcs.wcs.ctype[0] == 'KX' and wcs.wcs.ctype[1] == 'KY':
        ft_wcs.wcs.ctype[0] = template_wcs.wcs.ctype[0]
        ft_wcs.wcs.ctype[1] = template_wcs.wcs.ctype[1]
    elif wcs.wcs.ctype[0] == 'UU_AZELGEO' and wcs.wcs.ctype[1] == 'VV_AZELGEO':
        ft_wcs.wcs.ctype[0] = template_wcs.wcs.ctype[0]
        ft_wcs.wcs.ctype[1] = template_wcs.wcs.ctype[1]
   
    else:
        raise NotImplementedError("Cannot IFFT specified axes {0}, {1}".format(wcs.wcs.ctype[0], wcs.wcs.ctype[1]))

    ft_data = fft(gd["pixels"].data.astype('complex'))
    return create_image_from_array(ft_data, wcs=template_wcs,
                                   polarisation_frame=gd.griddata_acc.polarisation_frame)


def pad_image(im: Image, shape):
    """Pad an image to desired shape, adding equally to all edges

    Appropriate for standard 4D image with axes (freq, pol, y, x). Only pads in y, x

    The wcs crpix is adjusted appropriately.

    :param im: Image to be padded
    :param shape: Shape in 4 dimensions
    :return: Padded image
    """
    
    if im["pixels"].data.shape == shape:
        return im
    else:
        newwcs = copy.deepcopy(im.image_acc.wcs)
        newwcs.wcs.crpix[0] = im.image_acc.wcs.wcs.crpix[0] + shape[3] // 2 - im["pixels"].data.shape[3] // 2
        newwcs.wcs.crpix[1] = im.image_acc.wcs.wcs.crpix[1] + shape[2] // 2 - im["pixels"].data.shape[2] // 2
        
        for axis, _ in enumerate(im["pixels"].data.shape):
            if shape[axis] < im["pixels"].data.shape[axis]:
                raise ValueError("Padded shape %s is smaller than input shape %s" % (shape, im["pixels"].data.shape))
        
        newdata = numpy.zeros(shape, dtype=im["pixels"].dtype)
        ystart = shape[2] // 2 - im["pixels"].data.shape[2] // 2
        yend = ystart + im["pixels"].data.shape[2]
        xstart = shape[3] // 2 - im["pixels"].data.shape[3] // 2
        xend = xstart + im["pixels"].data.shape[3]
        newdata[..., ystart:yend, xstart:xend] = im["pixels"][...]
        return create_image_from_array(newdata, newwcs, polarisation_frame=im.image_acc.polarisation_frame)


def create_w_term_like(im: Image, w, phasecentre=None, remove_shift=False, dopol=False) -> Image:
    """Create an image with a w term phase term in it:

    .. math::

        I(l,m) = e^{-2 \\pi j (w(\\sqrt{1-l^2-m^2}-1)}


    The phasecentre is used as the delay centre for the w term (i.e. where n==0)

    :param im: template image
    :param phasecentre: SkyCoord definition of phasecentre
    :param w: w value to evaluate
    :param remove_shift:
    :param dopol: Do screen in polarisation?
    :return: Image
    """
    
    #assert image_is_canonical(im)
    fim_shape = list(im["pixels"].data.shape)
    if not dopol:
        fim_shape[1] = 1
    
    wcs = im.image_acc.wcs
    fim_array = numpy.zeros(fim_shape, dtype='complex')
    cellsize = abs(wcs.wcs.cdelt[0]) * numpy.pi / 180.0
    nchan, npol, _, npixel = fim_shape
    if phasecentre is SkyCoord:
        wcentre = phasecentre.to_pixel(wcs, origin=0)
    else:
        wcentre = [wcs.wcs.crpix[0] - 1.0, wcs.wcs.crpix[1] - 1.0]
    
    fim_array[...] = w_beam(npixel, npixel * cellsize, w=w, cx=wcentre[0], cy=wcentre[1],
                                 remove_shift=remove_shift)[numpy.newaxis, numpy.newaxis, ...]

    fim = create_image_from_array(fim_array, wcs=wcs, polarisation_frame=im.image_acc.polarisation_frame)

    fov = npixel * cellsize
    fresnel = numpy.abs(w) * (0.5 * fov) ** 2
    log.debug('create_w_term_image: For w = %.1f, field of view = %.6f, Fresnel number = %.2f' % (w, fov, fresnel))
    
    return fim


def scale_and_rotate_image(im, angle=0.0, scale=None, order=5):
    """ Scale and then rotate and image in x, y axes

    Applies scale then rotates

    :param im: Image
    :param angle: Angle in radians
    :param scale: Scale [scale_x, scale_y]
    :param order: Order of interpolation (0-5)
    :return:
    """
    from scipy.ndimage.interpolation import affine_transform
    
    nchan, npol, ny, nx = im["pixels"].data.shape
    c_in = 0.5 * numpy.array([ny, nx])
    c_out = 0.5 * numpy.array([ny, nx])
    rot = numpy.array([[numpy.cos(angle), -numpy.sin(angle)],
                       [numpy.sin(angle), numpy.cos(angle)]])
    inv_rot = rot.T
    if scale is None:
        scale = [1.0, 1.0]
    
    newim = create_empty_image_like(im)
    inv_scale = numpy.diag(scale)
    inv_transform = numpy.dot(inv_scale, inv_rot)
    offset = c_in - numpy.dot(inv_transform, c_out)
    for chan in range(nchan):
        for pol in range(npol):
            if im["pixels"].data.dtype == "complex":
                newim["pixels"].data[chan, pol] = affine_transform(im["pixels"].data[chan, pol].real,
                                                         inv_transform,
                                                         offset=offset,
                                                         order=order,
                                                         output_shape=(ny, nx)).astype("float") + \
                                        1.0j * affine_transform(im["pixels"].data[chan, pol].imag,
                                                                inv_transform,
                                                                offset=offset,
                                                                order=order,
                                                                output_shape=(ny, nx)).astype("float")
            elif im["pixels"].data.dtype == "float":
                newim["pixels"].data[chan, pol] = affine_transform(im["pixels"].data[chan, pol].real,
                                                         inv_transform,
                                                         offset=offset,
                                                         order=order,
                                                         output_shape=(ny, nx)).astype("float")
            else:
                raise ValueError("Cannot process data type {}".format(im["pixels"].data.dtype))
    
    return newim


def rotate_image(im, angle=0.0, order=5):
    """ Rotate an image in x, y axes

    :param im: Image
    :param angle: Angle in radians
    :param order: Order of interpolation (0-5)
    :return:
    """
    
    from scipy.ndimage.interpolation import rotate
    newim = im.copy(deep=True)
    if newim["pixels"].data.dtype == "complex":
        newim["pixels"].data = rotate(im["pixels"].data.real, angle=numpy.rad2deg(angle), axes=(-2, -1), order=order) + \
                     1j * rotate(im["pixels"].data.imag, angle=numpy.rad2deg(angle), axes=(-2, -1), order=order)
    else:
        newim["pixels"].data = rotate(im["pixels"].data, angle=numpy.rad2deg(angle), axes=(-2, -1), order=order)
    return newim


def apply_voltage_pattern_to_image(im: Image, vp: Image, inverse=False, min_det=1e-1, **kwargs) -> Image:
    """Apply a voltage pattern to an image

    For each pixel, the application is as follows:

    I_{corrected}(l,m) = vp(l,m) I(l,m) jones(j,m).H

    :param im: Image to have jones applied
    :param vp: Jones image to be applied
    :param inverse: Apply the inverse (default=False)
    :param min_det: Minimum determinant to correct
    :return: new Image with Jones applied
    """
    
    #assert image_is_canonical(im)
    
    #assert isinstance(im, Image)
    #assert isinstance(vp, Image)
    
    newim = create_empty_image_like(im)
    
    if inverse:
        log.debug('apply_gaintable: Apply inverse voltage pattern image')
    else:
        log.debug('apply_gaintable: Apply voltage pattern image')
    
    is_scalar = vp.image_acc.shape[1] == 1
    
    nchan, npol, ny, nx = im["pixels"].data.shape
    
    assert im["pixels"].data.shape == vp["pixels"].data.shape
    
    if is_scalar:
        log.debug('apply_voltage_pattern_to_image: Scalar voltage pattern')
        if inverse:
            for chan in range(nchan):
                pb = (vp["pixels"].data[chan, 0, ...] * numpy.conjugate(vp["pixels"].data[chan, 0, ...])).real
                newim["pixels"].data[chan, 0, ...] *= pb
        else:
            for chan in range(nchan):
                pb = (vp["pixels"].data[chan, 0, ...] * numpy.conjugate(vp["pixels"].data[chan, 0, ...])).real
                mask = pb > 0.0
                newim["pixels"].data[chan, 0, ...][mask] /= pb[mask]
    else:
        log.debug('apply_voltage_pattern_to_image: Full Jones voltage pattern')
        polim = convert_stokes_to_polimage(im, vp.image_acc.polarisation_frame)
        assert npol == 4
        im_t = numpy.transpose(polim["pixels"].data, (0, 2, 3, 1)).reshape([nchan, ny, nx, 2, 2])
        vp_t = numpy.transpose(vp["pixels"].data, (0, 2, 3, 1)).reshape([nchan, ny, nx, 2, 2])
        newim_t = numpy.zeros([nchan, ny, nx, 2, 2], dtype='complex')
        for chan in range(nchan):
            for y in range(ny):
                for x in range(nx):
                    newim_t[chan, y, x] = apply_jones(vp_t[chan, y, x], im_t[chan, y, x], inverse, min_det=min_det)
        
        newim = create_image_from_array(newim_t.reshape([nchan, ny, nx, 4]).transpose((0, 3, 1, 2)),
                                        wcs=im.image_acc.wcs,
                                        polarisation_frame=vp.image_acc.polarisation_frame)
        newim = convert_polimage_to_stokes(newim)
        
        return newim
