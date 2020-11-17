""" Image deconvolution functions

The standard deconvolution algorithms are provided:

    hogbom: Hogbom CLEAN See: Hogbom CLEAN A&A Suppl, 15, 417, (1974)
    
    msclean: MultiScale CLEAN See: Cornwell, T.J., Multiscale CLEAN (IEEE Journal of Selected Topics in Sig Proc,
    2008 vol. 2 pp. 793-801)

    mfsmsclean: MultiScale Multi-Frequency See: U. Rau and T. J. Cornwell, “A multi-scale multi-frequency
    deconvolution algorithm for synthesis imaging in radio interferometry,” A&A 532, A71 (2011).

For example to make dirty image and PSF, deconvolve, and then restore::

    model = create_image_from_visibility(vt, cellsize=0.001, npixel=256)
    dirty, sumwt = invert_2d(vt, model)
    psf, sumwt = invert_2d(vt, model, dopsf=True)

    comp, residual = deconvolve_cube(dirty, psf, niter=1000, threshold=0.001, fracthresh=0.01, window_shape='quarter',
                                 gain=0.7, algorithm='msclean', scales=[0, 3, 10, 30])

    restored = restore_cube(comp, psf, residual)

"""

__all__ = ['deconvolve_cube', 'restore_cube', 'fit_psf']

import logging
import warnings

import numpy
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.modeling import models, fitting

from rascil.data_models.memory_data_models import Image
from rascil.data_models.parameters import get_parameter
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.arrays.cleaners import hogbom, hogbom_complex, msclean, msmfsclean
from rascil.processing_components.image.operations import calculate_image_frequency_moments, \
    calculate_image_from_frequency_moments, image_is_canonical
from rascil.processing_components.image.operations import create_image_from_array

# from photutils import fit_2dgaussian

# warnings.simplefilter('ignore', AstropyDeprecationWarning)

log = logging.getLogger('rascil-logger')


def deconvolve_cube(dirty: Image, psf: Image, prefix='', **kwargs) -> (Image, Image):
    """ Clean using a variety of algorithms
    
    The algorithms available are:
    
    hogbom: Hogbom CLEAN See: Hogbom CLEAN A&A Suppl, 15, 417, (1974)

    hogbom-complex: Complex Hogbom CLEAN of stokesIQUV image
    
    msclean: MultiScale CLEAN See: Cornwell, T.J., Multiscale CLEAN (IEEE Journal of Selected Topics in Sig Proc,
    2008 vol. 2 pp. 793-801)

    mfsmsclean, msmfsclean, mmclean: MultiScale Multi-Frequency See: U. Rau and T. J. Cornwell,
    “A multi-scale multi-frequency deconvolution algorithm for synthesis imaging in radio interferometry,” A&A 532,
    A71 (2011).
    
    For example::
    
        comp, residual = deconvolve_cube(dirty, psf, niter=1000, gain=0.7, algorithm='msclean',
                                         scales=[0, 3, 10, 30], threshold=0.01)
                                         
    For the MFS clean, the psf must have number of channels >= 2 * nmoment
    
    :param dirty: Image dirty image
    :param psf: Image Point Spread Function
    :param window_shape: Window image (Bool) - clean where True
    :param mask: Window in the form of an image, overrides window_shape
    :param algorithm: Cleaning algorithm: 'msclean'|'hogbom'|'mfsmsclean'
    :param gain: loop gain (float) 0.7
    :param threshold: Clean threshold (0.0)
    :param fractional_threshold: Fractional threshold (0.01)
    :param scales: Scales (in pixels) for multiscale ([0, 3, 10, 30])
    :param nmoment: Number of frequency moments (default 3)
    :param findpeak: Method of finding peak in mfsclean: 'Algorithm1'|'ASKAPSoft'|'CASA'|'RASCIL', Default is RASCIL.
    :return: component image, residual image

    See also
        :py:func:`rascil.processing_components.arrays.cleaners.hogbom`
        :py:func:`rascil.processing_components.arrays.cleaners.hogbom_complex`
        :py:func:`rascil.processing_components.arrays.cleaners.msclean`
        :py:func:`rascil.processing_components.arrays.cleaners.msmfsclean`

    """
    
    # assert isinstance(dirty, Image), dirty
    assert image_is_canonical(dirty)
    # assert isinstance(psf, Image), psf
    assert image_is_canonical(psf)
    
    window_shape = get_parameter(kwargs, 'window_shape', None)
    if window_shape == 'quarter':
        log.info("deconvolve_cube %s: window is inner quarter" % prefix)
        qx = dirty["pixels"].shape[3] // 4
        qy = dirty["pixels"].shape[2] // 4
        window = numpy.zeros_like(dirty["pixels"].data)
        window[..., (qy + 1):3 * qy, (qx + 1):3 * qx] = 1.0
        log.info('deconvolve_cube %s: Cleaning inner quarter of each sky plane' % prefix)
    elif window_shape == 'no_edge':
        edge = get_parameter(kwargs, 'window_edge', 16)
        nx = dirty["pixels"].shape[3]
        ny = dirty["pixels"].shape[2]
        window = numpy.zeros_like(dirty["pixels"].data)
        window[..., (edge + 1):(ny - edge), (edge + 1):(nx - edge)] = 1.0
        log.info('deconvolve_cube %s: Window omits %d-pixel edge of each sky plane' % (prefix, edge))
    elif window_shape is None:
        log.info("deconvolve_cube %s: Cleaning entire image" % prefix)
        window = None
    else:
        raise ValueError("Window shape %s is not recognized" % window_shape)
    
    mask = get_parameter(kwargs, 'mask', None)
    if isinstance(mask, Image):
        if window is not None:
            log.warning('deconvolve_cube %s: Overriding window_shape with mask image' % (prefix))
        window = mask["pixels"].data
    
    psf_support = get_parameter(kwargs, 'psf_support',
                                max(dirty["pixels"].shape[2] // 2, dirty["pixels"].shape[3] // 2))
    if (psf_support <= psf["pixels"].shape[2] // 2) and ((psf_support <= psf["pixels"].shape[3] // 2)):
        centre = [psf["pixels"].shape[2] // 2, psf["pixels"].shape[3] // 2]
        psf = psf.isel(x=slice((centre[0] - psf_support), (centre[0] + psf_support)),
                       y=slice((centre[1] - psf_support), (centre[1] + psf_support)))
        log.info('deconvolve_cube %s: PSF support = +/- %d pixels' % (prefix, psf_support))
        log.info('deconvolve_cube %s: PSF shape %s' % (prefix, str(psf["pixels"].data.shape)))
    else:
        log.info("Using entire psf for dconvolution")
    
    algorithm = get_parameter(kwargs, 'algorithm', 'msclean')
    
    if algorithm == 'msclean':
        log.info("deconvolve_cube %s: Multi-scale clean of each polarisation and channel separately" %
                 prefix)
        gain = get_parameter(kwargs, 'gain', 0.7)
        assert 0.0 < gain < 2.0, "Loop gain must be between 0 and 2"
        thresh = get_parameter(kwargs, 'threshold', 0.0)
        assert thresh >= 0.0
        niter = get_parameter(kwargs, 'niter', 100)
        assert niter > 0
        scales = get_parameter(kwargs, 'scales', [0, 3, 10, 30])
        fracthresh = get_parameter(kwargs, 'fractional_threshold', 0.01)
        assert 0.0 < fracthresh < 1.0
        
        comp_array = numpy.zeros_like(dirty["pixels"].data)
        residual_array = numpy.zeros_like(dirty["pixels"].data)
        for channel in range(dirty["pixels"].data.shape[0]):
            for pol in range(dirty["pixels"].data.shape[1]):
                if psf["pixels"].data[channel, pol, :, :].max():
                    log.info("deconvolve_cube %s: Processing pol %d, channel %d" % (prefix, pol, channel))
                    if window is None:
                        comp_array[channel, pol, :, :], residual_array[channel, pol, :, :] = \
                            msclean(dirty["pixels"].data[channel, pol, :, :], psf["pixels"].data[channel, pol, :, :],
                                    None, gain, thresh, niter, scales, fracthresh, prefix)
                    else:
                        comp_array[channel, pol, :, :], residual_array[channel, pol, :, :] = \
                            msclean(dirty["pixels"].data[channel, pol, :, :], psf["pixels"].data[channel, pol, :, :],
                                    window[channel, pol, :, :], gain, thresh, niter, scales, fracthresh,
                                    prefix)
                else:
                    log.info("deconvolve_cube %s: Skipping pol %d, channel %d" % (prefix, pol, channel))
        
        comp_image = create_image_from_array(comp_array, dirty.wcs, dirty.polarisation_frame)
        residual_image = create_image_from_array(residual_array, dirty.wcs, dirty.polarisation_frame)
    
    elif algorithm == 'msmfsclean' or algorithm == 'mfsmsclean' or algorithm == 'mmclean':
        findpeak = get_parameter(kwargs, "findpeak", 'RASCIL')
        
        log.info("deconvolve_cube %s: Multi-scale multi-frequency clean of each polarisation separately"
                 % prefix)
        nmoment = get_parameter(kwargs, "nmoment", 3)
        assert nmoment >= 1, "Number of frequency moments must be greater than or equal to one"
        nchan = dirty["pixels"].shape[0]
        assert nchan > 2 * (nmoment - 1), "Require nchan %d > 2 * (nmoment %d - 1)" % (nchan, 2 * (nmoment - 1))
        dirty_taylor = calculate_image_frequency_moments(dirty, nmoment=nmoment)
        if nmoment > 1:
            log.debug(psf)
            log.debug(nmoment)
            psf_taylor = calculate_image_frequency_moments(psf, nmoment=2 * nmoment)
        else:
            psf_taylor = calculate_image_frequency_moments(psf, nmoment=1)
        psf_peak = numpy.max(psf_taylor["pixels"].data)
        dirty_taylor["pixels"].data /= psf_peak
        psf_taylor["pixels"].data /= psf_peak
        log.info("deconvolve_cube %s: Shape of Dirty moments image %s" %
                 (prefix, str(dirty_taylor["pixels"].shape)))
        log.info("deconvolve_cube %s: Shape of PSF moments image %s" % (prefix, str(psf_taylor["pixels"].shape)))
        gain = get_parameter(kwargs, 'gain', 0.7)
        assert 0.0 < gain < 2.0, "Loop gain must be between 0 and 2"
        thresh = get_parameter(kwargs, 'threshold', 0.0)
        assert thresh >= 0.0
        niter = get_parameter(kwargs, 'niter', 100)
        assert niter > 0
        scales = get_parameter(kwargs, 'scales', [0, 3, 10, 30])
        fracthresh = get_parameter(kwargs, 'fractional_threshold', 0.1)
        assert 0.0 < fracthresh < 1.0
        
        comp_array = numpy.zeros(dirty_taylor["pixels"].data.shape)
        residual_array = numpy.zeros(dirty_taylor["pixels"].data.shape)
        for pol in range(dirty_taylor["pixels"].data.shape[1]):
            # Always use the Stokes I PSF
            if psf_taylor["pixels"].data[0, 0, :, :].max():
                log.info("deconvolve_cube %s: Processing pol %d" % (prefix, pol))
                if window is None:
                    comp_array[:, pol, :, :], residual_array[:, pol, :, :] = \
                        msmfsclean(dirty_taylor["pixels"].data[:, pol, :, :],
                                   psf_taylor["pixels"].data[:, 0, :, :],
                                   None, gain, thresh, niter, scales, fracthresh, findpeak, prefix)
                else:
                    log.info('deconvolve_cube %s: Clean window has %d valid pixels'
                             % (prefix, int(numpy.sum(window[0, pol]))))
                    comp_array[:, pol, :, :], residual_array[:, pol, :, :] = \
                        msmfsclean(dirty_taylor["pixels"].data[:, pol, :, :], psf_taylor["pixels"].data[:, 0, :, :],
                                   window[0, pol, :, :], gain, thresh, niter, scales, fracthresh,
                                   findpeak, prefix)
            else:
                log.info("deconvolve_cube %s: Skipping pol %d" % (prefix, pol))
        
        comp_image = create_image_from_array(comp_array, dirty_taylor.wcs, dirty.polarisation_frame)
        residual_image = create_image_from_array(residual_array, dirty_taylor.wcs, dirty.polarisation_frame)
        
        return_moments = get_parameter(kwargs, "return_moments", False)
        if not return_moments:
            log.info("deconvolve_cube %s: calculating spectral cubes" % prefix)
            comp_image = calculate_image_from_frequency_moments(dirty, comp_image)
            residual_image = calculate_image_from_frequency_moments(dirty, residual_image)
        else:
            log.info("deconvolve_cube %s: constructed moment cubes" % prefix)
    
    elif algorithm == 'hogbom':
        log.info("deconvolve_cube %s: Hogbom clean of each polarisation and channel separately"
                 % prefix)
        gain = get_parameter(kwargs, 'gain', 0.1)
        assert 0.0 < gain < 2.0, "Loop gain must be between 0 and 2"
        thresh = get_parameter(kwargs, 'threshold', 0.0)
        assert thresh >= 0.0
        niter = get_parameter(kwargs, 'niter', 100)
        assert niter > 0
        fracthresh = get_parameter(kwargs, 'fractional_threshold', 0.1)
        assert 0.0 < fracthresh < 1.0
        
        comp_array = numpy.zeros(dirty["pixels"].data.shape)
        residual_array = numpy.zeros(dirty["pixels"].data.shape)
        for channel in range(dirty["pixels"].data.shape[0]):
            for pol in range(dirty["pixels"].data.shape[1]):
                if psf["pixels"].data[channel, pol, :, :].max():
                    log.info("deconvolve_cube %s: Processing pol %d, channel %d" % (prefix, pol, channel))
                    if window is None:
                        comp_array[channel, pol, :, :], residual_array[channel, pol, :, :] = \
                            hogbom(dirty["pixels"].data[channel, pol, :, :], psf["pixels"].data[channel, pol, :, :],
                                   None, gain, thresh, niter, fracthresh, prefix)
                    else:
                        comp_array[channel, pol, :, :], residual_array[channel, pol, :, :] = \
                            hogbom(dirty["pixels"].data[channel, pol, :, :], psf["pixels"].data[channel, pol, :, :],
                                   window[channel, pol, :, :], gain, thresh, niter, fracthresh, prefix)
                else:
                    log.info("deconvolve_cube %s: Skipping pol %d, channel %d" % (prefix, pol, channel))
        
        comp_image = create_image_from_array(comp_array, dirty.wcs, dirty.polarisation_frame)
        residual_image = create_image_from_array(residual_array, dirty.wcs, dirty.polarisation_frame)
    elif algorithm == 'hogbom-complex':
        log.info("deconvolve_cube_complex: Hogbom-complex clean of each channel separately")
        gain = get_parameter(kwargs, 'gain', 0.1)
        assert 0.0 < gain < 2.0, "Loop gain must be between 0 and 2"
        thresh = get_parameter(kwargs, 'threshold', 0.0)
        assert thresh >= 0.0
        niter = get_parameter(kwargs, 'niter', 100)
        assert niter > 0
        fracthresh = get_parameter(kwargs, 'fractional_threshold', 0.1)
        assert 0.0 <= fracthresh < 1.0
        
        comp_array = numpy.zeros(dirty["pixels"].data.shape)
        residual_array = numpy.zeros(dirty["pixels"].data.shape)
        for channel in range(dirty["pixels"].data.shape[0]):
            for pol in range(dirty["pixels"].data.shape[1]):
                if pol == 0 or pol == 3:
                    if psf["pixels"].data[channel, pol, :, :].max():
                        log.info("deconvolve_cube_complex: Processing pol %d, channel %d" % (pol, channel))
                        if window is None:
                            comp_array[channel, pol, :, :], residual_array[channel, pol, :, :] = \
                                hogbom(dirty["pixels"].data[channel, pol, :, :], psf["pixels"].data[channel, pol, :, :],
                                       None, gain, thresh, niter, fracthresh)
                        else:
                            comp_array[channel, pol, :, :], residual_array[channel, pol, :, :] = \
                                hogbom(dirty["pixels"].data[channel, pol, :, :], psf["pixels"].data[channel, pol, :, :],
                                       window[channel, pol, :, :], gain, thresh, niter, fracthresh)
                    else:
                        log.info("deconvolve_cube_complex: Skipping pol %d, channel %d" % (pol, channel))
                if pol == 1:
                    if psf["pixels"].data[channel, 1:2, :, :].max():
                        log.info("deconvolve_cube_complex: Processing pol 1 and 2, channel %d" % (channel))
                        if window is None:
                            comp_array[channel, 1, :, :], comp_array[channel, 2, :, :], residual_array[channel, 1, :,
                                                                                        :], residual_array[channel, 2,
                                                                                            :, :] = hogbom_complex(
                                dirty["pixels"].data[channel, 1, :, :], dirty["pixels"].data[channel, 2, :, :],
                                psf["pixels"].data[channel, 1, :, :],
                                psf["pixels"].data[channel, 2, :, :], None, gain, thresh, niter, fracthresh)
                        else:
                            comp_array[channel, 1, :, :], comp_array[channel, 2, :, :], residual_array[channel, 1, :,
                                                                                        :], residual_array[channel, 2,
                                                                                            :, :] = hogbom_complex(
                                dirty["pixels"].data[channel, 1, :, :], dirty["pixels"].data[channel, 2, :, :],
                                psf["pixels"].data[channel, 1, :, :],
                                psf["pixels"].data[channel, 2, :, :], window[channel, pol, :, :], gain, thresh, niter,
                                fracthresh)
                    else:
                        log.info("deconvolve_cube_complex: Skipping pol 1 and 2, channel %d" % (channel))
                if pol == 2:
                    continue
        
        comp_image = create_image_from_array(comp_array, dirty.wcs, polarisation_frame=PolarisationFrame('stokesIQUV'))
        residual_image = create_image_from_array(residual_array, dirty.wcs,
                                                 polarisation_frame=PolarisationFrame('stokesIQUV'))
    
    
    else:
        raise ValueError('deconvolve_cube %s: Unknown algorithm %s' % (prefix, algorithm))
    
    return comp_image, residual_image


def fit_psf(psf: Image, **kwargs):
    """ Fit PSF using astropy.modeling

    :params psf: Input PSF
    :return: fitted PSF, Gaussian2D, size

    """
    assert isinstance(psf, Image), psf
    assert image_is_canonical(psf)
    
    npixel = psf["pixels"].data.shape[3]
    sl = slice(npixel // 2 - 7, npixel // 2 + 8)
    y, x = numpy.mgrid[sl, sl]
    z = psf["pixels"].data[0, 0, sl, sl]
    
    # isotropic at the moment!
    from scipy.optimize import minpack
    try:
        p_init = models.Gaussian2D(amplitude=numpy.max(z), x_mean=numpy.mean(x), y_mean=numpy.mean(y))
        fit_p = fitting.LevMarLSQFitter()
        with warnings.catch_warnings():
            # Ignore model linearity warning from the fitter
            warnings.simplefilter('ignore')
            fit = fit_p(p_init, x, y, z)
        if fit.x_stddev <= 0.0 or fit.y_stddev <= 0.0:
            log.warning('fit_psf: error in fitting to psf, using 1 pixel stddev')
            beam_pixels = (1.0, 1.0, 0.0)
        else:
            # Note that the order here is minor, major, pa
            beam_pixels = (fit.x_stddev.value, fit.y_stddev.value, fit.theta.value)
            log.debug('fit_psf: fitted = {} pixels'.format(beam_pixels))
    except minpack.error as err:
        log.warning('fit_psf: minpack error, using 1 pixel stddev')
        beam_pixels = (1.0, 1.0, 0.0)
    except ValueError as err:
        log.warning('fit_psf: warning in fit to psf, using 1 pixel stddev')
        beam_pixels = (1.0, 1.0, 0.0)

    cellsize = 3600.0 * numpy.abs((psf["x"][0].data - psf["x"][-1].data)) / len(psf["x"])

    clean_beam = {"bmaj": beam_pixels[1] * cellsize,
                  "bmin": beam_pixels[0] * cellsize,
                  "bpa": beam_pixels[2]}

    return clean_beam


def restore_cube(model: Image, psf: Image, residual=None, **kwargs) -> Image:
    """ Restore the model image to the residuals

    :params psf: Input PSF
    :return: restored image

    """
    restored = model.copy(deep=True)
    
    cellsize = 3600.0 * numpy.abs((psf["x"][0].data - psf["x"][-1].data)) / len(psf["x"])
    
    clean_beam = get_parameter(kwargs, "cleanbeam", None)
    
    if clean_beam is None:
        clean_beam = fit_psf(psf)
        log.info('restore_cube: Using fitted clean beam = {}'.format(clean_beam))
    else:
        beam_pixels = [clean_beam["bmaj"] / cellsize, clean_beam["bmin"] / cellsize, clean_beam[["bpa"]]]
        log.info('restore_cube: Using specified clean beam = {}'.format(clean_beam))
    
    norm = 2.0 * numpy.pi * beam_pixels[0] * beam_pixels[1]
    gk = Gaussian2DKernel(x_stddev=beam_pixels[0], y_stddev=beam_pixels[1], theta=beam_pixels[2])
    # By convention, we normalise the peak not the integral so this is the volume of the Gaussian
    # norm = 2.0 * numpy.pi * size ** 2
    # gk = Gaussian2DKernel(size)
    for chan in range(model["pixels"].shape[0]):
        for pol in range(model["pixels"].shape[1]):
            restored["pixels"].data[chan, pol, :, :] = norm * convolve_fft(model["pixels"].data[chan, pol, :, :], gk,
                                                                           normalize_kernel=False, allow_huge=True)
    if residual is not None:
        restored["pixels"].data += residual["pixels"].data
    
    restored["pixels"].data = restored["pixels"].data.astype("float")
    
    return restored
