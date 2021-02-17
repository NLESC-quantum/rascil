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
from rascil.processing_components.skycomponent import copy_skycomponent
from rascil.processing_components.util.coordinate_support import skycoord_to_lmn
from rascil.processing_components.visibility.base import calculate_blockvisibility_phasor

log = logging.getLogger('rascil-logger')


def dft_skycomponent_visibility(vis: BlockVisibility, sc: Union[Skycomponent, List[Skycomponent]], **kwargs) \
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
    direction_cosines = list() # lmn vector for each component

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
        direction_cosine = numpy.array([l, m, numpy.sqrt(1 - l ** 2 - m ** 2) - 1.0])

        direction_cosines.append(direction_cosine)

    direction_cosines = numpy.array(direction_cosines)
    vfluxes = numpy.array(vfluxes)
    vis['vis'].data = dft_kernel(direction_cosines, vfluxes, vis.uvw_lambda, **kwargs)

    return vis


cuda_kernel_source = r'''
#include <cupy/complex.cuh>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define INDEX_3D(N3, N2, N1, I3, I2, I1)         (N1 * (N2 * I3 + I2) + I1)
#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, I1) (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)

extern "C" {

__global__ void dft_kernel(
        const int num_components,
        const int num_pols,
        const int num_channels,
        const int num_baselines,
        const int num_times,
        const double3         *const __restrict__ direction_cosines,        // Source direction cosines [num_components]
        const complex<double> *const __restrict__ vfluxes,    // Source fluxes [num_components, num_channels, num_pols]
        const double3         *const __restrict__ uvw_lambda, // UVW in lambda [num_times, num_baselines, num_channels]
        complex<double>       *__restrict__ vis)              // Visibilities  [num_times, num_baselines, num_channels, num_pols]
{
    // Local (per-thread) visibility.
    complex<double> vis_local[4]; // Allow up to 4 polarisations.
    vis_local[0] = vis_local[1] = vis_local[2] = vis_local[3] = 0.0;

    // Get indices of the output array this thread is working on.
    const int i_baseline = blockDim.x * blockIdx.x + threadIdx.x;
    const int i_channel  = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_time     = blockDim.z * blockIdx.z + threadIdx.z;

    // Bounds check.
    if (num_pols > 4 ||
            i_baseline >= num_baselines ||
            i_channel >= num_channels ||
            i_time >= num_times) {
        return;
    }

    // Load uvw-coordinates.
    const double3 uvw = uvw_lambda[INDEX_3D(
            num_times, num_baselines, num_channels,
            i_time, i_baseline, i_channel)];

    // Loop over components and calculate phase for each.
    for (int i_component = 0; i_component < num_components; ++i_component) {
        double sin_phase, cos_phase;
        const double3 dir = direction_cosines[i_component];
        const double phase = -2.0 * M_PI * (
                dir.x * uvw.x + dir.y * uvw.y + dir.z * uvw.z);
        sincos(phase, &sin_phase, &cos_phase);
        complex<double> phasor(cos_phase, sin_phase);

        // Multiply by flux in each polarisation and accumulate.
        const unsigned int i_pol_start = INDEX_3D(
                num_components, num_channels, num_pols,
                i_component, i_channel, 0);
        if (num_pols == 1) {
            vis_local[0] += (phasor * vfluxes[i_pol_start]);
        } else if (num_pols == 4) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                vis_local[i] += (phasor * vfluxes[i_pol_start + i]);
            }
        }
    }

    // Write out local visibility.
    for (int i = 0; i < num_pols; ++i) {
        const unsigned int i_out = INDEX_4D(num_times, num_baselines,
                num_channels, num_pols, i_time, i_baseline, i_channel, i);
        vis[i_out] = vis_local[i];
    }
}

}
'''


def dft_kernel(direction_cosines, vfluxes, uvw_lambda, dft_compute_kernel=None):
    """ CPU computational kernel for DFT

    :param direction_cosines: Direction cosines [ncomp, 3]
    :param vfluxes: Fluxes [ncomp, nchan, npol]
    :param uvw_lambda: UVW in lambda [ntimes, nbaselines, nchan, 3]
    :return: Vis [ntimes, nbaselines, nchan, npol]
    """

    if dft_compute_kernel is None:
        dft_compute_kernel = "cpu_einsum"

    if dft_compute_kernel == "gpu_cupy_einsum":
        import cupy
        with cupy.cuda.Device(0):
            uvw_lambda_gpu = cupy.asarray(uvw_lambda.data)
            direction_cosines_gpu = cupy.asarray(direction_cosines)
            vfluxes_gpu = cupy.asarray(vfluxes)
            phasors_gpu = \
                cupy.exp(-2j * numpy.pi * cupy.einsum("tbfs,cs->ctbf", uvw_lambda_gpu, direction_cosines_gpu))[..., cupy.newaxis]
            sum_gpu = cupy.sum(vfluxes_gpu[:, cupy.newaxis, cupy.newaxis, ...] * phasors_gpu, axis=0)
            return cupy.asnumpy(sum_gpu)
    elif dft_compute_kernel == "gpu_cupy_raw":
        import cupy

        # Get the dimension sizes.
        (num_times, num_baselines, num_channels, _) = uvw_lambda.shape
        (num_components, _, num_pols) = vfluxes.shape

        # Get a handle to the GPU kernel.
        module = cupy.RawModule(code=cuda_kernel_source)
        kernel_dft = module.get_function("dft_kernel")

        # Allocate GPU memory and copy input arrays.
        direction_cosines_gpu = cupy.asarray(direction_cosines)
        fluxes_gpu = cupy.asarray(vfluxes)
        uvw_gpu = cupy.asarray(uvw_lambda)
        vis_gpu = cupy.zeros((num_times, num_baselines, num_channels, num_pols),
                             dtype=cupy.complex128
        )

        # Define GPU kernel parameters, thread block size and grid size.
        num_threads = (128, 2, 2)  # Product must not exceed 1024.
        num_blocks = (
            (num_baselines + num_threads[0] - 1) // num_threads[0],
            (num_channels + num_threads[1] - 1) // num_threads[1],
            (num_times + num_threads[2] - 1) // num_threads[2]
        )
        args = (
            num_components, num_pols, num_channels, num_baselines, num_times,
            direction_cosines_gpu, fluxes_gpu, uvw_gpu, vis_gpu
        )

        # Call the GPU kernel and copy results to host.
        kernel_dft(num_blocks, num_threads, args)
        return cupy.asnumpy(vis_gpu)
    elif dft_compute_kernel == "cpu_einsum":
        phasors = \
           numpy.exp(-2j * numpy.pi * numpy.einsum("tbfs,cs->ctbf", uvw_lambda.data, direction_cosines))[..., numpy.newaxis]
        return numpy.sum(vfluxes[:, numpy.newaxis, numpy.newaxis, ...] * phasors, axis=0)
    elif dft_compute_kernel == "cpu_numpy":
        phasors = \
        numpy.exp(-2j * numpy.pi * numpy.sum(uvw_lambda.data
                                             * direction_cosines[:, numpy.newaxis, numpy.newaxis, numpy.newaxis, :], axis=-1))[..., numpy.newaxis]
        return numpy.sum(vfluxes[:, numpy.newaxis, numpy.newaxis, ...] * phasors, axis=0)
    elif dft_compute_kernel == "cpu_unrolled":
        phasors = \
            numpy.exp(
                -2j * numpy.pi * (uvw_lambda.data[..., 0] * direction_cosines[:, numpy.newaxis, numpy.newaxis, numpy.newaxis, 0] +
                                  uvw_lambda.data[..., 1] * direction_cosines[:, numpy.newaxis, numpy.newaxis, numpy.newaxis, 1] +
                                  uvw_lambda.data[..., 2] * direction_cosines[:, numpy.newaxis, numpy.newaxis, numpy.newaxis, 2]))[
                ..., numpy.newaxis]
        return numpy.sum(vfluxes[:, numpy.newaxis, numpy.newaxis, ...] * phasors, axis=0)
    elif dft_compute_kernel == "cpu_looped":
        ncomp, _ = direction_cosines.shape
        ntimes, nbaselines, nchan, _ = uvw_lambda.shape
        npol = vfluxes.shape[-1]
        vis = numpy.zeros([ntimes, nbaselines, nchan, npol], dtype='complex')
        for icomp in range(ncomp):
            phasor = numpy.exp(-2j * numpy.pi * numpy.sum(uvw_lambda.data * direction_cosines[icomp, :], axis=-1))
            for pol in range(npol):
                vis[..., pol] += vfluxes[icomp, :, pol] * phasor
        return vis
    else:
        raise ValueError(f"dft_compute_kernel {dft_compute_kernel} not known")



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
