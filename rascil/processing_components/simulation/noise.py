"""
Functions that add noise.

"""

__all__ = ['calculate_noise_blockvisibility', 'addnoise_visibility']

import logging

import numpy

from rascil.data_models.memory_data_models import BlockVisibility

log = logging.getLogger('rascil-logger')

def calculate_noise_blockvisibility(bandwidth, int_time, diameter, t_sys, eta):
    """Calculate noise rms per visibility [nchan, npol]

    :param bandwidth: (Hz)
    :param int_time: Integration time (s)
    :param diameter: Diameter (m)
    :param t_sys: (K)
    :param eta: Efficiency
    :returns: Sigma [nrows, nchan]
    """
    
    k_b = 1.38064852e-23
    area = numpy.pi * (diameter / 2.) ** 2
    bt = numpy.outer(int_time, bandwidth)
    sigma = (numpy.sqrt(2) * k_b * t_sys) / (area * eta * (numpy.sqrt(bt)))
    sigma *= 1e26
    return sigma


def addnoise_visibility(vis, t_sys=None, eta=None):
    """ Add noise to a visibility
    
    TODO: Obtain sensitivity values from vis as a function of frequency
    
    :param vis:
    :param t_sys: System temperature
    :param eta: Efficiency
    :return: vis with noise added
    """
    assert isinstance(vis, BlockVisibility), vis
    
    if t_sys is None:
        t_sys = 20.0
    
    if eta is None:
        eta = 0.78
    
    sigma = calculate_noise_blockvisibility(vis.channel_bandwidth.values, vis.data['integration_time'].data,
                                            vis.configuration.diameter.data[0], t_sys=t_sys, eta=eta)
    log.debug('addnoise_visibility: RMS noise value (first integration, first channel): %g' % sigma[0, 0])
    ntimes, nbaseline, nchan, npol = vis.vis.shape
    shape = (nbaseline, npol)
    for time in range(ntimes):
        for chan in range(nchan):
            vis.data["vis"].data[time, ..., chan, :].real += numpy.random.normal(0, sigma[time, chan], shape)
            vis.data["vis"].data[time, ..., chan, :].imag += numpy.random.normal(0, sigma[time, chan], shape)
    return vis
