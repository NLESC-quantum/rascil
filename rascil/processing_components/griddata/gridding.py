""" Imaging is based on used of the FFT to perform Fourier transforms efficiently. Since the observed visibility data_models
do not arrive naturally on grid points, the sampled points are resampled on the FFT grid using a convolution function to
smear out the sample points. The resulting grid points are then FFT'ed. The result can be corrected for the griddata
convolution function by division in the image plane of the transform.

This module contains functions for performing the griddata process and the inverse degridding process.

The GridData data model is used to hold the specification of the desired result.

GridData, ConvolutionFunction and Vis/BlockVis always have the same PolarisationFrame. Conversion to
stokesIQUV is only done in the image plane.
"""

__all__ = ['convolution_mapping_blockvisibility',
           'grid_blockvisibility_to_griddata',
           'degrid_blockvisibility_from_griddata',
           'fft_griddata_to_image',
           'fft_image_to_griddata',
           'griddata_merge_weights',
           'grid_blockvisibility_weight_to_griddata',
           'griddata_blockvisibility_reweight']

import logging

import astropy.constants as constants
import numpy
import numpy.testing

from rascil.data_models.memory_data_models import BlockVisibility, GridData, ConvolutionFunction, \
    Image
from rascil.processing_components.fourier_transforms import ifft, fft
from rascil.processing_components.griddata.operations import copy_griddata
from rascil.processing_components.image.operations import create_image_from_array
from rascil.processing_components.visibility.base import copy_visibility

log = logging.getLogger('rascil-logger')


def convolution_mapping_blockvisibility(vis, griddata, chan, cf, channel_tolerance=1e-8):
    """Find the mappings between visibility, griddata, and convolution function

    :param vis:
    :param griddata:
    :param cf:
    :param channel_tolerance:
    :return:
    """

    ##assert isinstance(vis, BlockVisibility), vis
    ##assert isinstance(griddata, GridData)
    ##assert isinstance(cf, ConvolutionFunction)
    assert vis.blockvisibility_acc.polarisation_frame == griddata.polarisation_frame

    u = vis.uvw_lambda.data[..., chan, 0].flat
    v = vis.uvw_lambda.data[..., chan, 1].flat
    w = vis.uvw_lambda.data[..., chan, 2].flat

    u = numpy.nan_to_num(u)
    v = numpy.nan_to_num(v)
    w = numpy.nan_to_num(w)

    pu_grid, pu_offset, pv_grid, pv_offset, pwc_fraction, pwc_grid, pwg_fraction, pwg_grid = \
        spatial_mapping(cf, griddata, u, v, w)

    return pu_grid, pu_offset, pv_grid, pv_offset, pwg_grid, pwg_fraction, pwc_grid, pwc_fraction


def spatial_mapping(cf, griddata, u, v, w):
    """ Map u,v,w per row into coordinates in the grid
    
    :param cf:
    :param griddata:
    :return:
    """

    ##assert isinstance(griddata, GridData)
    ##assert isinstance(cf, ConvolutionFunction)
    assert cf.polarisation_frame == griddata.polarisation_frame

    numpy.testing.assert_almost_equal(griddata.grid_wcs.wcs.cdelt[0], cf.cf_wcs.wcs.cdelt[0], 7)
    numpy.testing.assert_almost_equal(griddata.grid_wcs.wcs.cdelt[1], cf.cf_wcs.wcs.cdelt[1], 7)
    ####### UV mapping
    # We use the grid_wcs's to do the coordinate conversion
    # Find the nearest grid points
    
    pu_grid, pv_grid = \
        numpy.round(griddata.grid_wcs.sub([1, 2]).wcs_world2pix(u, v, 0)).astype('int')
    assert numpy.min(pu_grid) >= 0, "image sampling wrong: U axis underflows: %f" % numpy.min(pu_grid)
    assert numpy.max(pu_grid) < griddata.griddata_acc.shape[3], "U axis overflows: %f" % numpy.max(pu_grid)
    assert numpy.min(pv_grid) >= 0, "image sampling wrong: V axis underflows: %f" % numpy.min(pv_grid)
    assert numpy.max(pv_grid) < griddata.griddata_acc.shape[4], "V axis overflows: %f" % numpy.max(pv_grid)
    # We now have the location of grid points, convert back to uv space and find the remainder (in wavelengths). We
    # then use this to calculate the subsampling indices (DUU, DVV)
    wu_grid, wv_grid = griddata.grid_wcs.sub([1, 2]).wcs_pix2world(pu_grid, pv_grid, 0)
    wu_subsample, wv_subsample = u - wu_grid, v - wv_grid
    pu_offset, pv_offset = \
        numpy.round(cf.cf_wcs.sub([3, 4]).wcs_world2pix(wu_subsample, wv_subsample, 0)).astype('int')
    assert numpy.min(pu_offset) >= 0, "image sampling wrong: DU axis underflows: %f" % numpy.min(pu_offset)
    assert numpy.max(pu_offset) < cf["pixels"].data.shape[3], "DU axis overflows: %f" % numpy.max(pu_offset)
    assert numpy.min(pv_offset) >= 0, "image sampling wrong: DV axis underflows: %f" % numpy.min(pv_offset)
    assert numpy.max(pv_offset) < cf["pixels"].data.shape[4], "DV axis overflows: %f" % numpy.max(pv_offset)
    ###### W mapping for Grid
    # nchan, npol, w, v, u
    pwg_pixel = griddata.grid_wcs.sub([3]).wcs_world2pix(w, 0)[0]
    # Find the nearest grid point
    pwg_grid = numpy.round(pwg_pixel).astype('int')
    if numpy.min(pwg_grid) < 0:
        print(w[0:10])
        print(cf.cf_wcs.sub([5]).__repr__())
    assert numpy.min(pwg_grid) >= 0, "W axis underflows: %f" % numpy.min(pwg_grid)
    assert numpy.max(pwg_grid) < cf["pixels"].data.shape[2], "W axis overflows: %f" % numpy.max(pwg_grid)
    pwg_fraction = pwg_pixel - pwg_grid
    ###### W mapping for CF
    # nchan, npol, w, dv, du, v, u
    pwc_pixel = cf.cf_wcs.sub([5]).wcs_world2pix(w, 0)[0]
    pwc_grid = numpy.round(pwc_pixel).astype('int')
    if numpy.min(pwc_grid) < 0:
        print(w[0:10])
        print(cf.cf_wcs.sub([5]).__repr__())
    assert numpy.min(pwc_grid) >= 0, "W axis underflows: %f" % numpy.min(pwc_grid)
    assert numpy.max(pwc_grid) < cf["pixels"].data.shape[2], "W axis overflows: %f" % numpy.max(pwc_grid)
    pwc_fraction = pwc_pixel - pwc_grid
    return pu_grid, pu_offset, pv_grid, pv_offset, pwc_fraction, pwc_grid, pwg_fraction, pwg_grid


def grid_blockvisibility_to_griddata(vis, griddata, cf):
    """Grid BlockVisibility onto a GridData

    :param vis: blockvisibility to be gridded
    :param griddata: GridData
    :param cf: Convolution function
    :return: GridData
    """

    ##assert isinstance(vis, BlockVisibility), vis
    ##assert isinstance(griddata, GridData)
    ##assert isinstance(cf, ConvolutionFunction)
    assert vis.blockvisibility_acc.polarisation_frame == griddata.polarisation_frame

    griddata["pixels"].data[...] = 0.0

    vis_to_im = numpy.round(
        griddata.grid_wcs.sub([5]).wcs_world2pix(vis.frequency.data, 0)[0]).astype('int')

    nrows, nbaselines, nvchan, nvpol = vis["vis"].data.shape
    nichan, nipol, _, _, _ = griddata["pixels"].data.shape

    fvist = numpy.nan_to_num(vis.blockvisibility_acc.flagged_vis.data.reshape([nrows * nbaselines, nvchan, nvpol]).T)
    fwtt = numpy.nan_to_num(vis.blockvisibility_acc.flagged_imaging_weight.data.reshape([nrows * nbaselines, nvchan, nvpol]).T)
    # Do this in place to avoid creating a new copy. Doing the conjugation outside the loop
    # reduces run time immensely
    ccf = numpy.conjugate(cf["pixels"].data)
    ccf = numpy.nan_to_num(ccf)
    _, _, _, _, _, gv, gu = ccf.shape
    du = gu // 2
    dv = gv // 2

    sumwt = numpy.zeros([nichan, nipol])
    
    gd = griddata["pixels"].data
    
    for vchan in range(nvchan):
        imchan = vis_to_im[vchan]
        pu_grid, pu_offset, pv_grid, pv_offset, pwg_grid, pwg_fraction, pwc_grid, pwc_fraction = \
            convolution_mapping_blockvisibility(vis, griddata, vchan, cf)
        for pol in range(nvpol):
            for row in range(nrows * nbaselines):
                subcf = ccf[imchan,
                        pol,
                        pwc_grid[row],
                        pv_offset[row],
                        pu_offset[row],
                        :, :]
                gd[imchan, \
                pol, \
                pwg_grid[row], \
                (pv_grid[row] - dv):(pv_grid[row] + dv), \
                (pu_grid[row] - du):(pu_grid[row] + du)] \
                    += subcf * fvist[pol, vchan, row] * fwtt[pol, vchan, row]
                sumwt[imchan, pol] += fwtt[pol, vchan, row]
    
    griddata["pixels"].data = numpy.nan_to_num(gd)
    return griddata, numpy.nan_to_num(sumwt)


def grid_blockvisibility_weight_to_griddata(vis, griddata: GridData, cf):
    """Grid BlockVisibility weight onto a GridData

    :param vis: BlockVisibility to be gridded
    :param griddata: GridData
    :param cf: Convolution function
    :return: GridData
    """
    ##assert isinstance(vis, BlockVisibility), vis
    #assert isinstance(griddata, GridData)
    #assert isinstance(cf, ConvolutionFunction)
    assert vis.blockvisibility_acc.polarisation_frame == griddata.polarisation_frame
    assert cf.polarisation_frame == griddata.polarisation_frame


    nchan, npol, nw, ny, nx = griddata.griddata_acc.shape
    sumwt = numpy.zeros([nchan, npol])

    _, _, _, _, _, gv, gu = cf["pixels"].data.shape
    vis_to_im = numpy.round(
        griddata.grid_wcs.sub([5]).wcs_world2pix(vis.frequency.data, 0)[0]).astype('int')

    griddata["pixels"].data[...] = 0.0
    real_gd = numpy.real(griddata["pixels"].data)

    nrows, nbaselines, nvchan, nvpol = vis.vis.shape


    # Transpose to get row varying fastest
    fwtt = vis.blockvisibility_acc.flagged_imaging_weight.data.reshape([nrows * nbaselines, nvchan, nvpol]).T

    for vchan in range(nvchan):
        imchan = vis_to_im[vchan]
        pu_grid, pu_offset, pv_grid, pv_offset, pwg_grid, _, _, _ = \
            convolution_mapping_blockvisibility(vis, griddata, vchan, cf)
        for pol in range(nvpol):
            for row in range(nrows * nbaselines):
                real_gd[imchan, pol, pwg_grid[row], pv_grid[row], pu_grid[row]] += fwtt[
                    pol, vchan, row]
                sumwt[imchan, pol] += fwtt[pol, vchan, row]

    griddata["pixels"].data = real_gd.astype("complex")

    return griddata, sumwt

    
def griddata_merge_weights(gd_list, algorithm='uniform'):
    """ Merge weights into one grid
    
    :param gd_list:
    :param gd:
    :param algorithm:
    :return:
    """
    centre = len(gd_list) // 2
    gd = copy_griddata(gd_list[centre][0])
    sumwt = gd_list[centre][1]

    frequency = 0.0
    bandwidth = 0.0

    for i, g in enumerate(gd_list):
        if i != centre:
            gd["pixels"].data += g[0]["pixels"].data
            sumwt += g[1]
        frequency += g[0].grid_wcs.wcs.crval[4]
        bandwidth += g[0].grid_wcs.wcs.cdelt[4]

    gd.grid_wcs.wcs.cdelt[4] = bandwidth
    gd.grid_wcs.wcs.crval[4] = frequency / len(gd_list)
    return (gd, sumwt)


def griddata_blockvisibility_reweight(vis, griddata, cf, weighting="uniform", robustness=0.0):
    """Reweight blockvisibility weight using the weights in griddata

    :param weighting:
    :param vis: blockvisibility to be reweighted
    :param griddata: GridData holding gridded weights
    :param cf: Convolution function
    :return: BlockVisibility with imaging_weights corrected
    """
    #assert isinstance(vis, BlockVisibility), vis
    #assert isinstance(griddata, GridData)
    #assert isinstance(cf, ConvolutionFunction)
    assert vis.blockvisibility_acc.polarisation_frame == griddata.polarisation_frame
    assert cf.polarisation_frame == griddata.polarisation_frame
    
    assert weighting in ["natural", "uniform", "robust"], "Weighting {} not supported".format(weighting)
    
    real_gd = numpy.real(griddata["pixels"].data)
    
    vis_to_im = numpy.round(
        griddata.grid_wcs.sub([5]).wcs_world2pix(vis.frequency, 0)[0]).astype('int')
    
    nrows, nbaselines,nvchan, nvpol = vis.vis.shape
    fwtt = vis.blockvisibility_acc.flagged_imaging_weight.data.reshape([nrows * nbaselines, nvchan, nvpol]).T

    if weighting == "uniform":
        for pol in range(nvpol):
            for vchan in range(nvchan):
                imchan = vis_to_im[vchan]
                pu_grid, pu_offset, pv_grid, pv_offset, pwg_grid, pwg_fraction, pwc_grid, pwc_fraction = \
                    convolution_mapping_blockvisibility(vis, griddata, vchan, cf)
                for row in range(nrows * nbaselines):
                    wt = real_gd[imchan, pol, pwg_grid[row], pv_grid[row], pu_grid[row]]
                    if wt > 0.0:
                        fwtt[pol, vchan, row] /= wt
        
        vis['imaging_weight'].data[...] = fwtt.T.reshape([nrows, nbaselines, nvchan, nvpol])
    
    elif weighting == "robust":
        # Equation 3.15, 3.16 in Briggs thesis
        sumlocwt = numpy.sum(real_gd)
        sumwt = numpy.sum(vis.blockvisibility_acc.flagged_weight)
        f2 = (5.0 * numpy.power(10.0, -robustness))**2 * sumwt / sumlocwt
        for pol in range(nvpol):
            for vchan in range(nvchan):
                imchan = vis_to_im[vchan]
                pu_grid, pu_offset, pv_grid, pv_offset, pwg_grid, pwg_fraction, pwc_grid, pwc_fraction = \
                    convolution_mapping_blockvisibility(vis, griddata, vchan, cf)
                for row in range(nrows * nbaselines):
                    wt = real_gd[imchan, pol, pwg_grid[row], pv_grid[row], pu_grid[row]]
                    fwtt[pol, vchan, row] /= (1 + f2 * wt)
        
        vis['imaging_weight'].data[...] = fwtt.T.reshape([nrows, nbaselines, nvchan, nvpol])
        
    elif weighting == "natural":
        vis['imaging_weight'].data[...] = vis['weight'].data[...]
    
    return vis


def degrid_blockvisibility_from_griddata(vis, griddata, cf, **kwargs):
    """Degrid blockVisibility from a GridData

    :param vis: blockvisibility to be degridded
    :param griddata: GridData containing image
    :param cf: Convolution function (as GridData)
    :param kwargs:
    :return: BlockVisibility
    """
    #assert isinstance(vis, BlockVisibility), vis
    #assert isinstance(griddata, GridData)
    #assert isinstance(cf, ConvolutionFunction)
    assert vis.blockvisibility_acc.polarisation_frame == griddata.polarisation_frame
    assert cf.polarisation_frame == griddata.polarisation_frame

    newvis = copy_visibility(vis, zero=True)

    nchan, npol, nz, oversampling, _, support, _ = cf["pixels"].data.shape
    vis_to_im = numpy.round(
        griddata.grid_wcs.sub([5]).wcs_world2pix(vis.frequency.data, 0)[0]).astype('int')

    nrows, nbaselines, nvchan, nvpol = vis.vis.shape
    fvist = numpy.zeros([nvpol, nvchan, nrows * nbaselines], dtype='complex')

    _, _, _, _, _, gv, gu = cf["pixels"].data.shape

    du = gu // 2
    dv = gv // 2

    gd = griddata["pixels"].data
    scf = cf["pixels"].data
    for vchan in range(nvchan):
        imchan = vis_to_im[vchan]
        pu_grid, pu_offset, pv_grid, pv_offset, pwg_grid, pwg_fraction, pwc_grid, pwc_fraction = \
            convolution_mapping_blockvisibility(vis, griddata, vchan, cf)
        for pol in range(nvpol):
            for row in range(nrows * nbaselines):
                subgrid = gd[imchan, \
                          pol, \
                          pwg_grid[row], \
                          (pv_grid[row] - dv):(pv_grid[row] + dv), \
                          (pu_grid[row] - du):(pu_grid[row] + du)]
                subcf = scf[imchan,
                        pol,
                        pwc_grid[row],
                        pv_offset[row],
                        pu_offset[row],
                        :, :]
                fvist[pol, vchan, row] = numpy.einsum('ij,ij', subgrid, subcf)# / numpy.sum(subcf.real)

        # import matplotlib.pyplot as plt
        # plt.clf()
        # plt.plot(pu_offset[::10], numpy.abs(fvist[0, 0, ::10]), '.')
        # plt.title("U offset")
        # plt.show(block=False)
        # plt.clf()
        # plt.plot(pv_offset[::10], numpy.abs(fvist[0, 0, ::10]), '.')
        # plt.title("V offset")
        # plt.show(block=False)
        # plt.clf()
        # plt.plot(pu_offset[::10], pv_offset[::10], '.')
        # plt.title("U vs V offset")
        # plt.show(block=False)

    newvis["vis"].data[...] = fvist.T.reshape([nrows, nbaselines, nvchan, nvpol])

    return newvis


def fft_griddata_to_image(griddata, gcf=None):
    """ FFT griddata after applying gcf

    If imaginary is true the data array is complex

    :param griddata:
    :param gcf: Grid correction image
    :return:
    """
    #assert isinstance(griddata, GridData)

    projected = numpy.sum(griddata["pixels"].data, axis=2)
    ny, nx = projected.shape[-2], projected.shape[-1]

    if gcf is None:
        im_data = ifft(projected) * float(nx) * float(ny)
    else:
        im_data = ifft(projected) * gcf["pixels"].data * float(nx) * float(ny)

    return create_image_from_array(im_data, griddata.projection_wcs, griddata.polarisation_frame)


def fft_image_to_griddata(im, griddata, gcf=None):
    """Fill griddata with transform of im

    :param griddata:
    :param gcf: Grid correction image
    :return:
    """
    # chan, pol, z, u, v, w
    #assert isinstance(im, Image)
    #assert isinstance(griddata, GridData)
    assert im.image_acc.polarisation_frame == griddata.polarisation_frame

    if gcf is None:
        griddata["pixels"].data[:, :, :, ...] = fft(im["pixels"].data)[:, :, numpy.newaxis, ...]
    else:
        griddata["pixels"].data[:, :, :, ...] = fft(im["pixels"].data * \
                                                    gcf["pixels"].data)[:, :, numpy.newaxis, ...]

    return griddata
