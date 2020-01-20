""" Visibility operations

"""

__all__ = ['append_visibility', 'sort_visibility', 'concatenate_blockvisibility_frequency',
           'concatenate_visibility', 'sum_visibility', 'subtract_visibility', 'qa_visibility',
           'remove_continuum_blockvisibility', 'divide_visibility', 'integrate_visibility_by_channel',
           'convert_blockvisibility_to_stokes', 'convert_blockvisibility_to_stokes',
           'convert_blockvisibility_to_stokesI', 'convert_visibility_to_stokesI',
           'convert_visibility_to_stokes']

import logging
import warnings
from typing import Union

import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models.memory_data_models import BlockVisibility, Visibility, QA
from rascil.data_models.polarisation import convert_linear_to_stokes, convert_circular_to_stokesI, convert_linear_to_stokesI, \
    convert_circular_to_stokes, PolarisationFrame
from rascil.processing_components.imaging import get_frequency_map
from rascil.processing_components.util.coordinate_support import skycoord_to_lmn, simulate_point
from rascil.processing_components.visibility import copy_visibility

log = logging.getLogger(__name__)


def append_visibility(vis: Union[Visibility, BlockVisibility], othervis: Union[Visibility, BlockVisibility]) \
        -> Union[Visibility, BlockVisibility]:
    """Append othervis to vis
    
    :param vis:
    :param othervis:
    :return: Visibility vis + othervis
    """
    
    if vis is None:
        return othervis
    
    assert isinstance(vis, Visibility) or isinstance(vis, BlockVisibility), vis
    
    assert vis.polarisation_frame == othervis.polarisation_frame, "Polarisation frames differ"
    assert abs(vis.phasecentre.ra.value - othervis.phasecentre.ra.value) < 1e-15, "RAs differ"
    assert abs(vis.phasecentre.dec.value - othervis.phasecentre.dec.value) < 1e-15, "Declinations differ"
    assert vis.phasecentre.separation(othervis.phasecentre).value < 1e-15, "Phasecentres differ"
    assert vis.source == othervis.source, "Not the same source"
    if isinstance(vis, BlockVisibility):
        assert numpy.max(numpy.abs(vis.frequency - othervis.frequency)) < 1e-6
    
    vis.data = numpy.hstack((vis.data, othervis.data))
    return vis


def sort_visibility(vis, order=None):
    """ Sort a visibility on a given column
    
    :param vis:
    :param order: Array of string of column to be used for sortin
    :return:
    """
    if order is None:
        order = ['index']
    vis.data = numpy.sort(vis.data, order=order)
    return vis


def concatenate_visibility(vis_list, sort=True):
    """Concatenate a list of visibilities, with an optional sort back to index order

    :param vis_list: List of vis
    :return: Concatendated visibility
    """
    if isinstance(vis_list, Visibility) or isinstance(vis_list, BlockVisibility):
        return vis_list
    
    assert len(vis_list) > 0
    
    vis = None
    for v in vis_list:
        if vis is None:
            vis = v
        else:
            assert v.polarisation_frame == vis.polarisation_frame
            assert v.phasecentre.separation(vis.phasecentre).value < 1e-15
            vis.data = numpy.hstack((vis.data, v.data))
    
    assert vis is not None
    
    if sort:
        vis = sort_visibility(vis, ['index'])
    
    return vis


def concatenate_blockvisibility_frequency(bvis_list):
    """Concatenate a list of BlockVisibility's in frequency

    The list should be in sequence of channels

    :param bvis_list: List of BlockVisibility
    :return: BlockVisibility
    """
    
    assert len(bvis_list) > 0
    
    nvis = bvis_list[0].nvis
    time = bvis_list[0].time
    frequency = numpy.array(numpy.array([bvis.frequency for bvis in bvis_list]).flat)
    channel_bandwidth = numpy.array(numpy.array([bvis.channel_bandwidth for bvis in bvis_list]).flat)
    nchan = len(frequency)
    ntimes, nants, _, _, npol = bvis_list[0].vis.shape
    uvw = bvis_list[0].uvw
    integration_time = bvis_list[0].integration_time
    vis = numpy.zeros([nvis, nants, nants, nchan, npol], dtype='complex')
    flags = numpy.ones([nvis, nants, nants, nchan, npol], dtype='int')
    weight = numpy.ones([nvis, nants, nants, nchan, npol])
    imaging_weight = numpy.ones([nvis, nants, nants, nchan, npol])
    
    echan = 0
    for ibv, bvis in enumerate(bvis_list):
        schan = echan
        echan = schan + len(bvis.frequency)
        vis[..., schan:echan, :] = bvis.vis[...]
        weight[..., schan:echan, :] = bvis.weight[...]
        flags[..., schan:echan, :] = bvis.flags[...]
        imaging_weight[..., schan:echan, :] = bvis.flagged_imaging_weight[...]
        vis[..., schan:echan, :] = bvis.vis[...]
    
    return BlockVisibility(vis=vis, flags=flags, weight=weight, imaging_weight=imaging_weight, uvw=uvw, time=time,
                           integration_time=integration_time, frequency=frequency, channel_bandwidth=channel_bandwidth,
                           polarisation_frame=bvis_list[0].polarisation_frame, source=bvis_list[0].source,
                           configuration=bvis_list[0].configuration, phasecentre=bvis_list[0].phasecentre,
                           meta=None)


def sum_visibility(vis: Visibility, direction: SkyCoord) -> numpy.array:
    """ Direct Fourier summation in a given direction

    :param vis: Visibility to be summed
    :param direction: Direction of summation
    :return: flux[nch,npol], weight[nch,pol]
    """
    # TODO: Convert to Visibility or remove?
    
    assert isinstance(vis, Visibility) or isinstance(vis, BlockVisibility), vis
    
    svis = copy_visibility(vis)
    
    l, m, n = skycoord_to_lmn(direction, svis.phasecentre)
    phasor = numpy.conjugate(simulate_point(svis.uvw, l, m))
    
    # Need to put correct mapping here
    _, frequency = get_frequency_map(svis, None)
    
    frequency = list(frequency)
    
    nchan = max(frequency) + 1
    npol = svis.polarisation_frame.npol
    
    flux = numpy.zeros([nchan, npol])
    weight = numpy.zeros([nchan, npol])
    
    coords = svis.vis, svis.flagged_weight, phasor, list(frequency)
    for v, wt, p, ic in zip(*coords):
        for pol in range(npol):
            flux[ic, pol] += numpy.real(wt[pol] * v[pol] * p)
            weight[ic, pol] += wt[pol]
    
    flux[weight > 0.0] = flux[weight > 0.0] / weight[weight > 0.0]
    flux[weight <= 0.0] = 0.0
    return flux, weight


def subtract_visibility(vis, model_vis, inplace=False):
    """ Subtract model_vis from vis, returning new visibility
    
    :param vis:
    :param model_vis:
    :return:
    """
    if isinstance(vis, Visibility):
        assert isinstance(model_vis, Visibility), model_vis
    elif isinstance(vis, BlockVisibility):
        assert isinstance(model_vis, BlockVisibility), model_vis
    else:
        raise RuntimeError("Types of vis and model visibility are invalid")
    
    assert vis.vis.shape == model_vis.vis.shape, "Observed %s and model visibilities %s have different shapes" \
                                                 % (vis.vis.shape, model_vis.vis.shape)
    
    if inplace:
        vis.data['vis'] = vis.data['vis'] - model_vis.data['vis']
        return vis
    else:
        residual_vis = copy_visibility(vis)
        residual_vis.data['vis'] = residual_vis.data['vis'] - model_vis.data['vis']
        return residual_vis


def qa_visibility(vis: Union[Visibility, BlockVisibility], context=None) -> QA:
    """Assess the quality of Visibility

    :param context:
    :param vis: Visibility to be assessed
    :return: QA
    """
    assert isinstance(vis, Visibility) or isinstance(vis, BlockVisibility), vis
    
    avis = numpy.abs(vis.vis)
    data = {'maxabs': numpy.max(avis),
            'minabs': numpy.min(avis),
            'rms': numpy.std(avis),
            'medianabs': numpy.median(avis)}
    qa = QA(origin='qa_visibility',
            data=data,
            context=context)
    return qa


def remove_continuum_blockvisibility(vis: BlockVisibility, degree=1, mask=None) -> BlockVisibility:
    """ Fit and remove continuum visibility

    Fit a polynomial in frequency of the specified degree where mask is True
  
    :param vis: BlockVisibility
    :param degree: Degree of polynomial
    :param mask: Mask of continuum
    :return: BlockVisibility
    """
    assert isinstance(vis, Visibility) or isinstance(vis, BlockVisibility), vis
    
    if mask is not None:
        assert numpy.sum(mask) > 2 * degree, "Insufficient channels for fit"
    
    nchan = len(vis.frequency)
    # TODO: optimise loop
    x = (vis.frequency - vis.frequency[nchan // 2]) / (vis.frequency[0] - vis.frequency[nchan // 2])
    for row in range(vis.nvis):
        for ant2 in range(vis.nants):
            for ant1 in range(vis.nants):
                for pol in range(vis.polarisation_frame.npol):
                    wt = numpy.sqrt(vis.flagged_weight[row, ant2, ant1, :, pol])
                    if mask is not None:
                        wt[mask] = 0.0
                    fit = numpy.polyfit(x, vis.data['vis'][row, ant2, ant1, :, pol], w=wt, deg=degree)
                    prediction = numpy.polyval(fit, x)
                    vis.data['vis'][row, ant2, ant1, :, pol] -= prediction
    return vis


def divide_visibility(vis: BlockVisibility, modelvis: BlockVisibility):
    """ Divide visibility by model forming visibility for equivalent point source

    This is a useful intermediate product for calibration. Variation of the visibility in time and
    frequency due to the model structure is removed and the data can be averaged to a limit determined
    by the instrumental stability. The weight is adjusted to compensate for the division.
    
    Zero divisions are avoided and the corresponding weight set to zero.

    :param vis:
    :param modelvis:
    :return:
    """
    assert isinstance(vis, Visibility) or isinstance(vis, BlockVisibility), vis
    
    # Different for scalar and vector/matrix cases
    isscalar = vis.polarisation_frame.npol == 1
    
    if isscalar:
        # Scalar case is straightforward
        x = numpy.zeros_like(vis.vis)
        xwt = numpy.abs(modelvis.vis) ** 2 * vis.flagged_weight
        mask = xwt > 0.0
        x[mask] = vis.vis[mask] / modelvis.vis[mask]
    else:
        nrows, nants, _, nchan, npol = vis.vis.shape
        nrec = 2
        assert nrec * nrec == npol
        xshape = (nrows, nants, nants, nchan, nrec, nrec)
        x = numpy.zeros(xshape, dtype='complex')
        xwt = numpy.zeros(xshape)
        # TODO: Remove filter when fixed to use ndarray
        warnings.simplefilter("ignore", category=PendingDeprecationWarning)
        
        # TODO: optimise loop
        for row in range(nrows):
            for ant1 in range(nants):
                for ant2 in range(ant1 + 1, nants):
                    for chan in range(nchan):
                        ovis = numpy.matrix(vis.vis[row, ant2, ant1, chan].reshape([2, 2]))
                        mvis = numpy.matrix(modelvis.vis[row, ant2, ant1, chan].reshape([2, 2]))
                        wt = numpy.matrix(vis.flagged_weight[row, ant2, ant1, chan].reshape([2, 2]))
                        x[row, ant2, ant1, chan] = numpy.matmul(numpy.linalg.inv(mvis), ovis)
                        xwt[row, ant2, ant1, chan] = numpy.dot(mvis, numpy.multiply(wt, mvis.H)).real
        x = x.reshape((nrows, nants, nants, nchan, nrec * nrec))
        xwt = xwt.reshape((nrows, nants, nants, nchan, nrec * nrec))
    
    pointsource_vis = BlockVisibility(data=None, frequency=vis.frequency, channel_bandwidth=vis.channel_bandwidth,
                                      phasecentre=vis.phasecentre, configuration=vis.configuration,
                                      uvw=vis.uvw, time=vis.time, integration_time=vis.integration_time, vis=x,
                                      weight=xwt, source=vis.source, meta=vis.meta)
    return pointsource_vis


def integrate_visibility_by_channel(vis: BlockVisibility) -> BlockVisibility:
    """ Integrate visibility across all channels, returning new visibility
    
    :param vis: BlockVisibility
    :return: BlockVisibility
    """
    
    assert isinstance(vis, Visibility) or isinstance(vis, BlockVisibility), vis
    
    vis_shape = list(vis.vis.shape)
    ntimes, nants, _, nchan, npol = vis_shape
    vis_shape[-2] = 1
    newvis = BlockVisibility(data=None,
                             frequency=numpy.ones([1]) * numpy.average(vis.frequency),
                             channel_bandwidth=numpy.ones([1]) * numpy.sum(vis.channel_bandwidth),
                             phasecentre=vis.phasecentre,
                             configuration=vis.configuration,
                             uvw=vis.uvw,
                             time=vis.time,
                             vis=numpy.zeros(vis_shape, dtype='complex'),
                             weight=numpy.zeros(vis_shape, dtype='float'),
                             imaging_weight=numpy.zeros(vis_shape, dtype='float'),
                             integration_time=vis.integration_time,
                             polarisation_frame=vis.polarisation_frame,
                             source=vis.source,
                             meta=vis.meta)
    
    newvis.data['vis'][..., 0, :] = numpy.sum(vis.data['vis'] * vis.flagged_weight, axis=-2)
    newvis.data['weight'][..., 0, :] = numpy.sum(vis.flagged_weight, axis=-2)
    newvis.data['imaging_weight'][..., 0, :] = numpy.sum(vis.flagged_imaging_weight, axis=-2)
    mask = newvis.flagged_weight > 0.0
    newvis.data['vis'][mask] = newvis.data['vis'][mask] / newvis.flagged_weight[mask]
    
    return newvis


def convert_visibility_to_stokes(vis):
    """Convert the polarisation frame data into Stokes parameters.

    :param vis: Visibility
    :return: Converted visibility data.
    """
    poldef = vis.polarisation_frame
    if poldef == PolarisationFrame('linear'):
        vis.data['vis'] = convert_linear_to_stokes(vis.data['vis'], polaxis=1)
        vis.polarisation_frame = PolarisationFrame('stokesIQUV')
    elif poldef == PolarisationFrame('circular'):
        vis.data['vis'] = convert_circular_to_stokes(vis.data['vis'], polaxis=1)
        vis.polarisation_frame = PolarisationFrame('stokesIQUV')
    return vis


def convert_blockvisibility_to_stokes(vis):
    """Convert the polarisation frame data into Stokes parameters.

    :param vis: Visibility
    :return: Converted visibility data.
    """
    poldef = vis.polarisation_frame
    if poldef == PolarisationFrame('linear'):
        vis.data['vis'] = convert_linear_to_stokes(vis.data['vis'], polaxis=4)
        vis.polarisation_frame = PolarisationFrame('stokesIQUV')
    elif poldef == PolarisationFrame('circular'):
        vis.data['vis'] = convert_circular_to_stokes(vis.data['vis'], polaxis=4)
        vis.polarisation_frame = PolarisationFrame('stokesIQUV')
    return vis


def convert_visibility_to_stokesI(vis):
    """Convert the polarisation frame data into Stokes I dropping other polarisations, return new Visibility

    :param vis: Visibility
    :return: Converted visibility data.
    """
    polarisation_frame = PolarisationFrame('stokesI')
    poldef = vis.polarisation_frame
    if poldef == PolarisationFrame('linear'):
        vis_data = convert_linear_to_stokesI(vis.data['vis'])
        vis_weight = (vis.flagged_weight[..., 0] + vis.flagged_weight[..., 3])[..., numpy.newaxis]
        vis_imaging_weight = (vis.flagged_imaging_weight[..., 0] + vis.flagged_imaging_weight[..., 3])[..., numpy.newaxis]
    elif poldef == PolarisationFrame('circular'):
        vis_data = convert_circular_to_stokesI(vis.data['vis'])
        vis_weight = (vis.flagged_weight[..., 0] + vis.flagged_weight[..., 3])[..., numpy.newaxis]
        vis_imaging_weight = (vis.flagged_imaging_weight[..., 0] + vis.flagged_imaging_weight[..., 3])[..., numpy.newaxis]
    else:
        raise NameError("Polarisation frame %s unknown" % poldef)
    
    return Visibility(frequency=vis.frequency, channel_bandwidth=vis.channel_bandwidth,
                      phasecentre=vis.phasecentre, configuration=vis.configuration, uvw=vis.uvw,
                      time=vis.time, antenna1=vis.antenna1, antenna2=vis.antenna2, vis=vis_data,
                      weight=vis_weight, imaging_weight=vis_imaging_weight, integration_time=vis.integration_time,
                      polarisation_frame=polarisation_frame, cindex=vis.cindex,
                      blockvis=vis.blockvis, source=vis.source, meta=vis.meta)


def convert_blockvisibility_to_stokesI(vis):
    """Convert the polarisation frame data into Stokes I dropping other polarisations, return new Visibility

    :param vis: Visibility
    :return: Converted visibility data.
   """
    polarisation_frame = PolarisationFrame('stokesI')
    poldef = vis.polarisation_frame
    if poldef == PolarisationFrame('linear'):
        vis_data = convert_linear_to_stokesI(vis.data['vis'])
        vis_weight = (vis.flagged_weight[..., 0] + vis.flagged_weight[..., 3])[..., numpy.newaxis]
        vis_imaging_weight = (vis.flagged_imaging_weight[..., 0] + vis.flagged_imaging_weight[..., 3])[..., numpy.newaxis]
    elif poldef == PolarisationFrame('circular'):
        vis_data = convert_circular_to_stokesI(vis.data['vis'])
        vis_weight = (vis.flagged_weight[..., 0] + vis.flagged_weight[..., 3])[..., numpy.newaxis]
        vis_imaging_weight = (vis.flagged_imaging_weight[..., 0] + vis.flagged_imaging_weight[..., 3])[..., numpy.newaxis]
    else:
        raise NameError("Polarisation frame %s unknown" % poldef)
    
    return BlockVisibility(frequency=vis.frequency, channel_bandwidth=vis.channel_bandwidth,
                           phasecentre=vis.phasecentre, configuration=vis.configuration, uvw=vis.uvw,
                           time=vis.time, vis=vis_data,
                           weight=vis_weight, imaging_weight=vis_imaging_weight, integration_time=vis.integration_time,
                           polarisation_frame=polarisation_frame, source=vis.source, meta=vis.meta)
