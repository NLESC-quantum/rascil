"""
Functions that add noise.

"""

__all__ = ['calculate_noise_blockvisibility', 'addnoise_visibility']

import logging

import numpy

from rascil.data_models.memory_data_models import BlockVisibility

log = logging.getLogger('logger')

def calculate_noise_blockvisibility(bandwidth, int_time, diameter, t_sys, eta):
    """Calculate noise rms per visibility

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
    
    sigma = calculate_noise_blockvisibility(vis.channel_bandwidth, vis.data['integration_time'],
                                            vis.configuration.diameter[0], t_sys=t_sys, eta=eta)
    log.debug('addnoise_visibility: RMS noise value (first integration, first channel): %g' % sigma[0, 0])
    for row in range(vis.nvis):
        for baseline in range(vis.baselines):
            for pol in range(vis.npol):
                vis.data["vis"][row, baseline, :, pol].real += numpy.random.normal(0, sigma[row, ...])
                vis.data["vis"][row, baseline, :, pol].imag += numpy.random.normal(0, sigma[row, ...])
    
    return vis
