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

__all__ = ['dft_skycomponent_visibility', 'idft_visibility_skycomponent']

import collections
import logging
from typing import List, Union

import numpy
from scipy import interpolate

from rascil.data_models.memory_data_models import BlockVisibility, Skycomponent
from rascil.data_models.polarisation import convert_pol_frame
from rascil.processing_components.util.coordinate_support import skycoord_to_lmn
from rascil.processing_components.skycomponent import copy_skycomponent
from rascil.processing_components.visibility.base import calculate_blockvisibility_phasor
import cupy

log = logging.getLogger('rascil-logger')


def dft_skycomponent_visibility(vis: BlockVisibility, sc: Union[Skycomponent, List[Skycomponent]]) \
        -> BlockVisibility:
    """DFT to get the visibility from a Skycomponent, for BlockVisibility

    :param vis: BlockVisibility
    :param sc: Skycomponent or list of SkyComponents
    :return: BlockVisibility or BlockVisibility
    """
    if sc is None:
        return vis

    if not isinstance(sc, collections.abc.Iterable):
        sc = [sc]

    vfluxes = list() # Flux for each component
    ses = list() # lmn vector for each component
    
    for comp in sc:
        #assert isinstance(comp, Skycomponent), comp
        flux = comp.flux
        if comp.polarisation_frame != vis.blockvisibility_acc.polarisation_frame:
            flux = convert_pol_frame(flux, comp.polarisation_frame, vis.blockvisibility_acc.polarisation_frame)

        # Interpolate in frequency if necessary
        if len(comp.frequency) == len(vis.frequency) and \
                numpy.allclose(comp.frequency,vis.frequency.data, rtol=1e-15):
            vflux = flux
        else:
            nchan, npol = flux.shape
            nvchan = len(vis.frequency)
            vflux = numpy.zeros([nvchan, npol])
            if nchan > 1:
                for pol in range(flux.shape[1]):
                    fint = interpolate.interp1d(comp.frequency, comp.flux[:, pol], kind="cubic")
                    vflux[:, pol] = fint(vis.frequency.data)
            else:
                # Just take the value since we cannot interpolate. Might want to put some
                # test here
                vflux = flux

        vfluxes.append(vflux)

        l, m, n = skycoord_to_lmn(comp.direction, vis.phasecentre)
        s = numpy.array([l, m, numpy.sqrt(1 - l ** 2 - m ** 2) - 1.0])
        
        ses.append(s)
    
    ses = numpy.array(ses)
    vfluxes = numpy.array(vfluxes)
    vis['vis'].data = dft_cpu_kernel(ses, vfluxes, vis.uvw_lambda)

    return vis


def dft_cpu_kernel(ses, vfluxes, uvw_lambda):
    """ CPU computational kernel for DFT
    
    :param ses: Direction cosines [ncomp, 3]
    :param vfluxes: Fluxes [ncomp, nchan, npol]
    :param uvw_lambda: UVW in lambda [ntimes, nbaselines, nchan]
    :return: Vis [ntimes, nbaselines, nchan, npol]
    """
    # with cupy.cuda.Device(0):
    #     uvw_lambda_gpu = cupy.asarray(uvw_lambda.data)
    #     ses_gpu = cupy.asarray(ses)
    #     vfluxes_gpu = cupy.asarray(vfluxes)
    #     phasors_gpu = \
    #         cupy.exp(-2j * numpy.pi * cupy.einsum("tbfs,cs->ctbf", uvw_lambda_gpu, ses_gpu))[..., cupy.newaxis]
    #     sum_gpu = cupy.sum(vfluxes_gpu[:, cupy.newaxis, cupy.newaxis, ...] * phasors_gpu, axis=0)
    #     return cupy.asnumpy(sum_gpu)
    phasors = \
        numpy.exp(-2j * numpy.pi * numpy.einsum("tbfs,cs->ctbf", uvw_lambda.data, ses))[..., numpy.newaxis]
    return numpy.sum(vfluxes[:, numpy.newaxis, numpy.newaxis, ...] * phasors, axis=0)


def idft_visibility_skycomponent(vis: BlockVisibility,
                                 sc: Union[Skycomponent, List[Skycomponent]]) -> \
        ([Skycomponent, List[Skycomponent]], List[numpy.ndarray]):
    """Inverse DFT a Skycomponent from BlockVisibility

    :param vis: BlockVisibility
    :param sc: Skycomponent or list of SkyComponents
    :return: Skycomponent or list of SkyComponents, array of weights
    """
    if sc is None:
        return sc

    if not isinstance(sc, collections.abc.Iterable):
        sc = [sc]

    newsc = list()
    weights_list = list()

    for comp in sc:
        #assert isinstance(comp, Skycomponent), comp
        newcomp = copy_skycomponent(comp)

        phasor = numpy.conjugate(calculate_blockvisibility_phasor(comp.direction, vis))
        flux = numpy.sum(vis.blockvisibility_acc.flagged_weight * vis.blockvisibility_acc.flagged_vis * phasor, axis=(0, 1))
        weight = numpy.sum(vis.blockvisibility_acc.flagged_weight, axis=(0, 1))

        flux[weight > 0.0] = flux[weight > 0.0] / weight[weight > 0.0]
        flux[weight <= 0.0] = 0.0
        if comp.polarisation_frame != vis.blockvisibility_acc.polarisation_frame:
            flux = convert_pol_frame(flux, vis.blockvisibility_acc.polarisation_frame, comp.polarisation_frame)

        newcomp.flux = flux

        newsc.append(newcomp)
        weights_list.append(weight)

    return newsc, weights_list


