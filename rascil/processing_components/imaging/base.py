"""
Functions that aid fourier transform processing. These are built on top of the core
functions in processing_components.fourier_transforms.

The measurement equation for a sufficently narrow field of view interferometer is:

.. math::

    V(u,v,w) =\\int I(l,m) e^{-2 \\pi j (ul+vm)} dl dm


The measurement equation for a wide field of view interferometer is:

.. math::

    V(u,v,w) =\\int \\frac{I(l,m)}{\\sqrt{1-l^2-m^2}} e^{-2 \\pi j (ul+vm + w(\\sqrt{1-l^2-m^2}-1))} dl dm

This and related modules contain various approachs for dealing with the wide-field problem where the
extra phase term in the Fourier transform cannot be ignored.
"""

__all__ = ['shift_vis_to_image',
           'normalize_sumwt',
           'predict_2d',
           'invert_2d',
           'predict_skycomponent_visibility',
           'create_image_from_visibility',
           'advise_wide_field',
           'visibility_recentre',
           'fill_blockvis_for_psf']

import logging
from typing import List, Union

import astropy.constants as constants
import astropy.units as units
import astropy.wcs as wcs
import numpy
from astropy.wcs.utils import pixel_to_skycoord

from rascil.data_models.memory_data_models import BlockVisibility, Image, Skycomponent
from rascil.data_models.parameters import get_parameter
from rascil.data_models.polarisation import PolarisationFrame, convert_pol_frame
from rascil.processing_components.griddata.gridding import  \
    grid_blockvisibility_to_griddata, fft_griddata_to_image, fft_image_to_griddata, \
    degrid_blockvisibility_from_griddata
from rascil.processing_components.griddata.kernels import create_pswf_convolutionfunction
from rascil.processing_components.griddata.operations import create_griddata_from_image
from rascil.processing_components.image import create_image_from_array, convert_polimage_to_stokes, \
    convert_stokes_to_polimage
from rascil.processing_components.visibility.base import copy_visibility, phaserotate_visibility

log = logging.getLogger('logger')


def shift_vis_to_image(vis: BlockVisibility, im: Image, tangent: bool = True, inverse: bool = False) \
        -> BlockVisibility:
    """Shift visibility in place to the phase centre of the Image

    :param vis: BlockVisibility
    :param im: Image model used to determine phase centre
    :param tangent: Is the shift purely on the tangent plane True|False
    :param inverse: Do the inverse operation True|False
    :return: visibility with phase shift applied and phasecentre updated

    """
    assert isinstance(vis, BlockVisibility), "vis is not a BlockVisibility: %r" % vis

    nchan, npol, ny, nx = im.data.shape

    # Convert the FFT definition of the phase center to world coordinates (1 relative)
    # This is the only place in RASCIL where the relationship between the image and visibility
    # frames is defined.

    image_phasecentre = pixel_to_skycoord(nx // 2 + 1, ny // 2 + 1, im.wcs, origin=1)
    if vis.phasecentre.separation(image_phasecentre).rad > 1e-15:
        if inverse:
            log.debug("shift_vis_from_image: shifting phasecentre from image phase centre %s to visibility phasecentre "
                      "%s" % (image_phasecentre, vis.phasecentre))
        else:
            log.debug("shift_vis_from_image: shifting phasecentre from vis phasecentre %s to image phasecentre %s" %
                      (vis.phasecentre, image_phasecentre))
        vis = phaserotate_visibility(vis, image_phasecentre, tangent=tangent, inverse=inverse)
        vis.phasecentre = im.phasecentre

    return vis


def normalize_sumwt(im: Image, sumwt) -> Image:
    """Normalize out the sum of weights

    The gridding weights are accumulated as a function of channel and polarisation. This function
    corrects for this sum of weights.

    :param im: Image, im.data has shape [nchan, npol, ny, nx]
    :param sumwt: Sum of weights [nchan, npol]
    """
    nchan, npol, _, _ = im.data.shape
    assert isinstance(im, Image), im
    assert sumwt is not None
    assert nchan == sumwt.shape[0]
    assert npol == sumwt.shape[1]
    for chan in range(nchan):
        for pol in range(npol):
            if sumwt[chan, pol] > 0.0:
                im.data[chan, pol, :, :] = im.data[chan, pol, :, :] / sumwt[chan, pol]
            else:
                im.data[chan, pol, :, :] = 0.0
    return im


def predict_2d(vis: BlockVisibility, model: Image, gcfcf=None, **kwargs) -> BlockVisibility:
    """ Predict using convolutional degridding.

    This is at the bottom of the layering i.e. all transforms are eventually expressed in terms of
    this function. Any shifting needed is performed here.

    :param vis: blockvisibility to be predicted
    :param model: model image
    :param gcfcf: (Grid correction function i.e. in image space, Convolution function i.e. in uv space)
    :return: resulting visibility (in place works)
    """

    if model is None:
        return vis

    assert isinstance(vis, BlockVisibility), vis

    _, _, ny, nx = model.data.shape

    if gcfcf is None:
        gcf, cf = create_pswf_convolutionfunction(model,
                                                  support=get_parameter(kwargs, "support", 8),
                                                  oversampling=get_parameter(kwargs, "oversampling", 127),
                                                  polarisation_frame=vis.polarisation_frame)
    else:
        gcf, cf = gcfcf

    griddata = create_griddata_from_image(model, polarisation_frame=vis.polarisation_frame)
    polmodel = convert_stokes_to_polimage(model, vis.polarisation_frame)
    griddata = fft_image_to_griddata(polmodel, griddata, gcf)
    vis = degrid_blockvisibility_from_griddata(vis, griddata=griddata, cf=cf)

    # Now we can shift the visibility from the image frame to the original visibility frame
    svis = shift_vis_to_image(vis, model, tangent=True, inverse=True)

    return svis


def invert_2d(vis: BlockVisibility, im: Image, dopsf: bool = False, normalize: bool = True,
              gcfcf=None, **kwargs) -> (Image, numpy.ndarray):
    """ Invert using 2D convolution function, using the specified convolution function

    Use the image im as a template. Do PSF in a separate call.

    This is at the bottom of the layering i.e. all transforms are eventually expressed in terms
    of this function. Any shifting needed is performed here.

    :param vis: blockvisibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :param normalize: Normalize by the sum of weights (True)
    :param gcfcf: (Grid correction function i.e. in image space, Convolution function i.e. in uv space)
    :return: resulting image

    """
    assert isinstance(vis, BlockVisibility), vis

    svis = copy_visibility(vis)

    if dopsf:
        svis = fill_blockvis_for_psf(svis)

    svis = shift_vis_to_image(svis, im, tangent=True, inverse=False)

    if gcfcf is None:
        gcf, cf = create_pswf_convolutionfunction(im,
                                                  support=get_parameter(kwargs, "support", 8),
                                                  oversampling=get_parameter(kwargs, "oversampling", 127),
                                                  polarisation_frame=vis.polarisation_frame)
    else:
        gcf, cf = gcfcf

    griddata = create_griddata_from_image(im, polarisation_frame=vis.polarisation_frame)
    griddata, sumwt = grid_blockvisibility_to_griddata(svis, griddata=griddata, cf=cf)
    result = fft_griddata_to_image(griddata, gcf)

    if normalize:
        result = normalize_sumwt(result, sumwt)

    result = convert_polimage_to_stokes(result, **kwargs)

    return result, sumwt


def fill_blockvis_for_psf(svis):
    """ Fill the visibility for calculation of PSF

    :param im:
    :param svis:
    :return: visibility with unit vis
    """
    if svis.polarisation_frame == PolarisationFrame("linear"):
        svis.data['vis'].values[..., 0] = 1.0 + 0.0j
        svis.data['vis'].values[..., 1:3] = 0.0 + 0.0j
        svis.data['vis'].values[..., 3] = 1.0 + 0.0j
    elif svis.polarisation_frame == PolarisationFrame("circular"):
        svis.data['vis'].values[..., 0] = 1.0 + 0.0j
        svis.data['vis'].values[..., 1:3] = 0.0 + 0.0j
        svis.data['vis'].values[..., 3] = 1.0 + 0.0j
    elif svis.polarisation_frame == PolarisationFrame("linearnp"):
        svis.data['vis'].values[...] = 1.0 + 0.0j
    elif svis.polarisation_frame == PolarisationFrame("circularnp"):
        svis.data['vis'].values[...] = 1.0 + 0.0j
    elif svis.polarisation_frame == PolarisationFrame("stokesI"):
        svis.data['vis'].values[...] = 1.0 + 0.0j
    else:
        raise ValueError("Cannot calculate PSF for {}".format(svis.polarisation_frame))
    
    return svis


def predict_skycomponent_visibility(vis: BlockVisibility,
                                    sc: Union[Skycomponent, List[Skycomponent]]) -> BlockVisibility:
    """Predict the visibility from a Skycomponent, add to existing visibility, for BlockVisibility

    Now replaced by dft_skycomponent_visibility

    :param vis: BlockVisibility
    :param sc: Skycomponent or list of SkyComponents
    :return: BlockVisibility or BlockVisibility
    """

    log.warning("predict_skycomponent_visibility: deprecated - please use dft_skycomponent_visibility")
    from rascil.processing_components.imaging.dft import dft_skycomponent_visibility
    return dft_skycomponent_visibility(vis, sc)


def create_image_from_visibility(vis: BlockVisibility, **kwargs) -> Image:
    """Make an empty image from params and BlockVisibility
    
    This makes an empty, template image consistent with the visibility, allowing optional overriding of select
    parameters. This is a convenience function and does not transform the visibilities.

    :param vis:
    :param phasecentre: Phasecentre (Skycoord)
    :param channel_bandwidth: Channel width (Hz)
    :param cellsize: Cellsize (radians)
    :param npixel: Number of pixels on each axis (512)
    :param frame: Coordinate frame for WCS (ICRS)
    :param equinox: Equinox for WCS (2000.0)
    :param nchan: Number of image channels (Default is 1 -> MFS)
    :return: image

    See also
        :py:func:`rascil.processing_components.image.operations.create_image`
    """
    assert isinstance(vis, BlockVisibility), \
        "vis is not a BlockVisibility: %r" % (vis)
    
    log.debug("create_image_from_visibility: Parsing parameters to get definition of WCS")

    imagecentre = get_parameter(kwargs, "imagecentre", vis.phasecentre)
    phasecentre = get_parameter(kwargs, "phasecentre", vis.phasecentre)

    # Spectral processing options
    ufrequency = numpy.unique(vis.frequency.values)
    frequency = get_parameter(kwargs, "frequency", vis.frequency.values)

    vnchan = len(ufrequency)

    inchan = get_parameter(kwargs, "nchan", vnchan)
    reffrequency = frequency[0] * units.Hz
    channel_bandwidth = get_parameter(kwargs, "channel_bandwidth", vis.channel_bandwidth.values[0]) * units.Hz


    if (inchan == vnchan) and vnchan > 1:
        log.debug(
            "create_image_from_visibility: Defining %d channel Image at %s, starting frequency %s, and bandwidth %s"
            % (inchan, imagecentre, reffrequency, channel_bandwidth))
    elif (inchan == 1) and vnchan > 1:
        assert numpy.abs(channel_bandwidth) > 0.0, "Channel width must be non-zero for mfs mode"
        log.debug("create_image_from_visibility: Defining single channel MFS Image at %s, starting frequency %s, "
                  "and bandwidth %s"
                  % (imagecentre, reffrequency, channel_bandwidth))
    elif inchan > 1 and vnchan > 1:
        assert numpy.abs(channel_bandwidth) > 0.0, "Channel width must be non-zero for mfs mode"
        log.debug("create_image_from_visibility: Defining multi-channel MFS Image at %s, starting frequency %s, "
                  "and bandwidth %s"
                  % (imagecentre, reffrequency, channel_bandwidth))
    elif (inchan == 1) and (vnchan == 1):
        assert numpy.abs(channel_bandwidth) > 0.0, "Channel width must be non-zero for mfs mode"
        log.debug("create_image_from_visibility: Defining single channel Image at %s, starting frequency %s, "
                  "and bandwidth %s"
                  % (imagecentre, reffrequency, channel_bandwidth))
    else:
        raise ValueError("create_image_from_visibility: unknown spectral mode inchan = {}, vnchan = {} "
                         .format(inchan, vnchan))

    # Image sampling options
    npixel = get_parameter(kwargs, "npixel", 512)
    if isinstance(vis, BlockVisibility):
        uvmax = numpy.max((numpy.abs(vis.uvw_lambda.values[..., 0:1])))
    else:
        uvmax = numpy.max((numpy.abs(vis.uvw[..., 0:1])))

    log.debug("create_image_from_visibility: uvmax = %f wavelengths" % uvmax)
    criticalcellsize = 1.0 / (uvmax * 2.0)
    log.debug("create_image_from_visibility: Critical cellsize = %f radians, %f degrees" % (
        criticalcellsize, criticalcellsize * 180.0 / numpy.pi))
    cellsize = get_parameter(kwargs, "cellsize", 0.5 * criticalcellsize)
    log.debug("create_image_from_visibility: Cellsize          = %g radians, %g degrees" % (cellsize,
                                                                                            cellsize * 180.0 / numpy.pi))
    override_cellsize = get_parameter(kwargs, "override_cellsize", True)
    if (override_cellsize and cellsize > criticalcellsize) or (cellsize == 0.0):
        log.debug("create_image_from_visibility: Resetting cellsize %g radians to criticalcellsize %g radians" % (
            cellsize, criticalcellsize))
        cellsize = criticalcellsize
    pol_frame = get_parameter(kwargs, "polarisation_frame", PolarisationFrame("stokesI"))
    inpol = pol_frame.npol

    # Now we can define the WCS, which is a convenient place to hold the info above
    # Beware of python indexing order! wcs and the array have opposite ordering
    shape = [inchan, inpol, npixel, npixel]
    log.debug("create_image_from_visibility: image shape is %s" % str(shape))
    w = wcs.WCS(naxis=4)
    # The negation in the longitude is needed by definition of RA, DEC
    w.wcs.cdelt = [-cellsize * 180.0 / numpy.pi, cellsize * 180.0 / numpy.pi, 1.0, channel_bandwidth.to(units.Hz).value]
    # The numpy definition of the phase centre of an FFT is n // 2 (0 - rel) so that's what we use for
    # the reference pixel. We have to use 0 rel everywhere.
    w.wcs.crpix = [npixel // 2 + 1, npixel // 2 + 1, 1.0, 1.0]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", 'STOKES', 'FREQ']
    w.wcs.crval = [phasecentre.ra.deg, phasecentre.dec.deg, 1.0, reffrequency.to(units.Hz).value]
    w.naxis = 4

    w.wcs.radesys = get_parameter(kwargs, 'frame', 'ICRS')
    w.wcs.equinox = get_parameter(kwargs, 'equinox', 2000.0)

    return create_image_from_array(numpy.zeros(shape), wcs=w, polarisation_frame=pol_frame)


def advise_wide_field(vis: BlockVisibility, delA=0.02,
                      oversampling_synthesised_beam=3.0,
                      guard_band_image=6.0, facets=1, wprojection_planes=1, verbose=True):
    """ Advise on parameters for wide field imaging.
    
    Calculate sampling requirements on various parameters
    
    For example::
    
        advice = advise_wide_field(vis, delA)
        wstep = get_parameter(kwargs, 'wstep', advice['w_sampling_primary_beam'])

    
    :param vis:
    :param delA: Allowed coherence loss (def: 0.02)
    :param oversampling_synthesised_beam: Oversampling of the synthesized beam (def: 3.0)
    :param guard_band_image: Number of primary beam half-widths-to-half-maximum to image (def: 6)
    :param facets: Number of facets on each axis
    :param wprojection_planes: Number of planes in wprojection
    :return: dict of advice
    """

    max_wavelength = constants.c.to('m s^-1').value / numpy.min(vis.frequency.values)
    if verbose:
        log.info("advise_wide_field: (max_wavelength) Maximum wavelength %.3f (meters)" % (max_wavelength))

    min_wavelength = constants.c.to('m s^-1').value / numpy.max(vis.frequency.values)
    if verbose:
        log.info("advise_wide_field: (min_wavelength) Minimum wavelength %.3f (meters)" % (min_wavelength))

    maximum_baseline = numpy.max(numpy.abs(vis.uvw.values)) / min_wavelength  # Wavelengths
    maximum_w = numpy.max(numpy.abs(vis.w.values)) / min_wavelength  # Wavelengths

    if verbose:
        log.info("advise_wide_field: (maximum_baseline) Maximum baseline %.1f (wavelengths)" % (maximum_baseline))
    assert maximum_baseline > 0.0, "Error in UVW coordinates: all uvw are zero"

    if verbose:
        log.info("advise_wide_field: (maximum_w) Maximum w %.1f (wavelengths)" % (maximum_w))

    diameter = numpy.min(vis.configuration.diameter.values)
    if verbose:
        log.info("advise_wide_field: (diameter) Station/dish diameter %.1f (meters)" % (diameter))
    assert diameter > 0.0, "Station/dish diameter must be greater than zero"

    primary_beam_fov = max_wavelength / diameter
    if verbose:
        log.info("advise_wide_field: (primary_beam_fov) Primary beam %s" % (rad_deg_arcsec(primary_beam_fov)))

    image_fov = primary_beam_fov * guard_band_image
    if verbose:
        log.info("advise_wide_field: (image_fov) Image field of view %s" % (rad_deg_arcsec(image_fov)))

    facet_fov = primary_beam_fov * guard_band_image / facets
    if facets > 1:
        if verbose:
            log.info("advise_wide_field: (facet_fov) Facet field of view %s" % (rad_deg_arcsec(facet_fov)))

    synthesized_beam = 1.0 / (maximum_baseline)
    if verbose:
        log.info("advise_wide_field: (synthesized_beam) Synthesized beam %s" % (rad_deg_arcsec(synthesized_beam)))

    cellsize = synthesized_beam / oversampling_synthesised_beam
    if verbose:
        log.info("advise_wide_field: (cellsize) Cellsize %s" % (rad_deg_arcsec(cellsize)))

    def pwr2(n):
        ex = numpy.ceil(numpy.log(n) / numpy.log(2.0)).astype('int')
        best = numpy.power(2, ex)
        return best

    def pwr23(n):
        ex = numpy.ceil(numpy.log(n) / numpy.log(2.0)).astype('int')
        best = numpy.power(2, ex)
        if best * 3 // 4 >= n:
            best = best * 3 // 4
        return best

    def pwr2345(n):
        # If pyfftw has been installed, next_fast_len would return the len of best performance
        try:
            import pyfftw
            best = pyfftw.next_fast_len(n)
        except ImportError:
            pyfftw = None
            number = numpy.array([2, 3, 4, 5])
            ex = numpy.ceil(numpy.log(n) / numpy.log(number)).astype('int')
            best = min(numpy.power(number[:], ex[:]))
        return best

    npixels = int(round(image_fov / cellsize))
    if verbose:
        log.info("advice_wide_field: (npixels) Npixels per side = %d" % (npixels))

    npixels2 = pwr2(npixels)
    if verbose:
        log.info("advice_wide_field: (npixels2) Npixels (power of 2) per side = %d" % (npixels2))

    npixels23 = pwr23(npixels)
    if verbose:
        log.info("advice_wide_field: (npixels23) Npixels (power of 2, 3) per side = %d" % (npixels23))

    npixels_min = pwr2345(npixels)
    if verbose:
        log.info("advice_wide_field: (npixels_min) Npixels (power of 2, 3, 4, 5) per side = %d" % (npixels_min))

    # Following equation is from Cornwell, Humphreys, and Voronkov (2012) (equation 24)
    # We will assume that the constraint holds at one quarter the entire FOV i.e. that
    # the full field of view includes the entire primary beam

    w_sampling_image = numpy.sqrt(2.0 * delA) / (numpy.pi * image_fov ** 2)
    if verbose:
        log.info("advice_wide_field: (w_sampling_image) W sampling for full image = %.1f (wavelengths)" % (w_sampling_image))

    if facets > 1:
        w_sampling_facet = numpy.sqrt(2.0 * delA) / (numpy.pi * facet_fov ** 2)
        if verbose:
            log.info("advice_wide_field: (w_sampling_facet) W sampling for facet = %.1f (wavelengths)" % (w_sampling_facet))
    else:
        w_sampling_facet = w_sampling_image

    w_sampling_primary_beam = numpy.sqrt(2.0 * delA) / (numpy.pi * primary_beam_fov ** 2)
    if verbose:
        log.info("advice_wide_field: (w_sampling_primary_beam) W sampling for primary beam = %.1f (wavelengths)" % (w_sampling_primary_beam))

    time_sampling_image = 86400.0 * (synthesized_beam / image_fov)
    if verbose:
        log.info("advice_wide_field: (time_sampling_image) Time sampling for full image = %.1f (s)" % (time_sampling_image))

    if facets > 1:
        time_sampling_facet = 86400.0 * (synthesized_beam / facet_fov)
        if verbose:
            log.info("advice_wide_field: (time_sampling_facet) Time sampling for facet = %.1f (s)" % (time_sampling_facet))

    time_sampling_primary_beam = 86400.0 * (synthesized_beam / primary_beam_fov)
    if verbose:
        log.info("advice_wide_field: (time_sampling_primary_beam) Time sampling for primary beam = %.1f (s)" % (time_sampling_primary_beam))

    max_freq = numpy.max(vis.frequency.values)
    
    freq_sampling_image = max_freq * (synthesized_beam / image_fov)
    if verbose:
        log.info("advice_wide_field: (freq_sampling_image) Frequency sampling for full image = %.1f (Hz)" % (freq_sampling_image))

    if facets > 1:
        freq_sampling_facet = max_freq * (synthesized_beam / facet_fov)
        if verbose:
            log.info("advice_wide_field: (freq_sampling_facet) Frequency sampling for facet = %.1f (Hz)" % (freq_sampling_facet))

    freq_sampling_primary_beam = max_freq * (synthesized_beam / primary_beam_fov)
    if verbose:
        log.info("advice_wide_field: (freq_sampling_primary_beam) Frequency sampling for primary beam = %.1f (Hz)" % (freq_sampling_primary_beam))

    wstep = w_sampling_primary_beam
    vis_slices = max(1, int(2 * maximum_w / wstep))
    wprojection_planes = vis_slices
    if verbose:
        log.info('advice_wide_field: (vis_slices) Number of planes in w stack %d (primary beam)' % (vis_slices))
        log.info('advice_wide_field: (wprojection_planes) Number of planes in w projection %d (primary beam)' % (
            wprojection_planes))

    nwpixels = int(2.0 * wprojection_planes * primary_beam_fov)
    nwpixels = nwpixels - nwpixels % 2
    if verbose:
        log.info('advice_wide_field: (nwpixels) W support = %d (pixels) (primary beam)' % nwpixels)

    del pwr2
    del pwr23
    return locals()


def rad_deg_arcsec(x):
    """ Stringify x in radian and degress forms
    
    """
    return "%.3g (rad) %.3g (deg) %.3g (asec)" % (x, 180.0 * x / numpy.pi, 3600.0 * 180.0 * x / numpy.pi)


def visibility_recentre(uvw, dl, dm):
    """ Compensate for kernel re-centering - see `w_kernel_function`.

    :param uvw: Visibility coordinates
    :param dl: Horizontal shift to compensate for
    :param dm: Vertical shift to compensate for
    :returns: Visibility coordinates re-centrered on the peak of their w-kernel
    """

    u, v, w = numpy.hsplit(uvw, 3)  # pylint: disable=unbalanced-tuple-unpacking
    return numpy.hstack([u - w * dl, v - w * dm, w])
