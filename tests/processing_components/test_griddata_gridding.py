""" Unit tests for image operations


"""
import functools
import logging
import os
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.griddata.gridding import fft_griddata_to_image, fft_image_to_griddata, \
    griddata_merge_weights, grid_blockvisibility_to_griddata, griddata_blockvisibility_reweight, \
    grid_blockvisibility_weight_to_griddata, degrid_blockvisibility_from_griddata
from rascil.processing_components.griddata.kernels import create_awterm_convolutionfunction, \
    create_pswf_convolutionfunction, create_box_convolutionfunction
from rascil.processing_components.griddata.operations import create_griddata_from_image
from rascil.processing_components.image.operations import export_image_to_fits, convert_stokes_to_polimage, \
    convert_polimage_to_stokes
from rascil.processing_components.image.operations import smooth_image
from rascil.processing_components.imaging import dft_skycomponent_visibility
from rascil.processing_components.imaging.base import normalize_sumwt
from rascil.processing_components.imaging.primary_beams import create_pb_generic
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.simulation import create_unittest_model, \
    create_unittest_components, ingest_unittest_visibility
from rascil.processing_components.skycomponent.operations import insert_skycomponent
from rascil.processing_components.visibility.operations import qa_visibility

log = logging.getLogger('rascil-logger')

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))



class TestGridDataGridding(unittest.TestCase):
    
    def setUp(self):
        from rascil.data_models.parameters import rascil_path
        self.dir = rascil_path('test_results')
        self.persist = os.getenv("RASCIL_PERSIST", False)
    
    def actualSetUp(self, zerow=True, image_pol=PolarisationFrame("stokesIQUV")):
        self.doplot = False
        self.npixel = 256
        self.cellsize = 0.0009
        self.low = create_named_configuration('LOWBD2', rmax=750.0)
        self.freqwin = 1
        self.vis_list = list()
        self.ntimes = 3
        self.times = numpy.linspace(-2.0, +2.0, self.ntimes) * numpy.pi / 12.0
        
        if self.freqwin == 1:
            self.frequency = numpy.array([1e8])
            self.channelwidth = numpy.array([4e7])
        else:
            self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
            self.channelwidth = numpy.array(self.freqwin * [self.frequency[1] - self.frequency[0]])
        
        self.image_pol = image_pol
        if image_pol == PolarisationFrame("stokesI"):
            self.vis_pol = PolarisationFrame("stokesI")
            f = numpy.array([100.0])
        elif image_pol == PolarisationFrame("stokesIQUV"):
            self.vis_pol = PolarisationFrame("linear")
            f = numpy.array([100.0, 20.0, -10.0, 1.0])
        elif image_pol == PolarisationFrame("stokesIQ"):
            self.vis_pol = PolarisationFrame("linearnp")
            f = numpy.array([100.0, 20.0])
        elif image_pol == PolarisationFrame("stokesIV"):
            self.vis_pol = PolarisationFrame("circularnp")
            f = numpy.array([100.0, 20.0])
        else:
            raise ValueError("Polarisation {} not supported".format(image_pol))
        
        flux = numpy.array([f * numpy.power(freq / 1e8, -0.7) for freq in self.frequency])
        
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.vis = ingest_unittest_visibility(self.low,
                                              self.frequency,
                                              self.channelwidth,
                                              self.times,
                                              self.vis_pol,
                                              self.phasecentre,
                                              zerow=zerow)
        
        self.model = create_unittest_model(self.vis, self.image_pol, cellsize=self.cellsize,
                                           npixel=self.npixel, nchan=self.freqwin)
        self.components = create_unittest_components(self.model, flux, applypb=False,
                                                     scale=0.5, single=False, symmetric=True)
        self.model = insert_skycomponent(self.model, self.components)
        
        self.vis = dft_skycomponent_visibility(self.vis, self.components)
        
        # Calculate the model convolved with a Gaussian.
        self.cmodel = smooth_image(self.model)
        if self.persist:
            export_image_to_fits(self.model, '%s/test_gridding_model.fits' % self.dir)
            export_image_to_fits(self.cmodel, '%s/test_gridding_cmodel.fits' % self.dir)
        pb = create_pb_generic(self.model, diameter=35.0, blockage=0.0, use_local=False)
        self.cmodel["pixels"].data *= pb["pixels"].data
        if self.persist:
            export_image_to_fits(self.cmodel, '%s/test_gridding_cmodel_pb.fits' % self.dir)
        self.peak = numpy.unravel_index(numpy.argmax(numpy.abs(self.cmodel["pixels"].data)),
                                        self.cmodel["pixels"].shape)
    
    def test_griddata_invert_pswf(self):
        self.actualSetUp(zerow=True)
        gcf, cf = create_pswf_convolutionfunction(self.model, polarisation_frame=self.vis_pol)
        griddata = create_griddata_from_image(self.model, polarisation_frame=self.vis_pol)
        griddata, sumwt = grid_blockvisibility_to_griddata(self.vis, griddata=griddata, cf=cf)
        cim = fft_griddata_to_image(griddata, gcf, gcf)
        cim = normalize_sumwt(cim, sumwt)
        im = convert_polimage_to_stokes(cim)
        if self.persist:
            export_image_to_fits(im, '%s/test_gridding_dirty_pswf.fits' % self.dir)
        self.check_peaks(im, 97.10594988, tol=1e-7)
    
    def test_griddata_invert_pswf_stokesIQ(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIQ"))
        gcf, cf = create_pswf_convolutionfunction(self.model, polarisation_frame=self.vis_pol)
        griddata = create_griddata_from_image(self.model, polarisation_frame=self.vis_pol)
        griddata, sumwt = grid_blockvisibility_to_griddata(self.vis, griddata=griddata, cf=cf)
        cim = fft_griddata_to_image(griddata, gcf, gcf)
        cim = normalize_sumwt(cim, sumwt)
        im = convert_polimage_to_stokes(cim)
        if self.persist:
            export_image_to_fits(im, '%s/test_gridding_dirty_pswf.fits' % self.dir)
        self.check_peaks(im, 97.10594988, tol=1e-7)
    
    def test_griddata_invert_aterm(self):
        self.actualSetUp(zerow=True)
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0, use_local=False)
        pb = make_pb(self.model)
        if self.persist:
            export_image_to_fits(pb, "%s/test_gridding_aterm_pb.fits" % self.dir)
        gcf, cf = create_awterm_convolutionfunction(self.model, make_pb=make_pb, nw=1, oversampling=16, support=16,
                                                    use_aaf=False, polarisation_frame=self.vis_pol)
        # cf_image = convert_convolutionfunction_to_image(cf)
        # cf_image.data = numpy.real(cf_image.data)
        # if self.persist:
        #     export_image_to_fits(cf_image, "%s/test_gridding_aterm_cf.fits" % self.dir)
        griddata = create_griddata_from_image(self.model, polarisation_frame=self.vis_pol)
        griddata, sumwt = grid_blockvisibility_to_griddata(self.vis, griddata=griddata, cf=cf)
        cim = fft_griddata_to_image(griddata, gcf, gcf)
        cim = normalize_sumwt(cim, sumwt)
        im = convert_polimage_to_stokes(cim)
        if self.persist:
            export_image_to_fits(im, '%s/test_gridding_dirty_aterm.fits' % self.dir)
        self.check_peaks(im, 97.10594988, tol=1e-7)
    
    def test_griddata_invert_aterm_noover(self):
        self.actualSetUp(zerow=True)
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0, use_local=False)
        pb = make_pb(self.model)
        if self.persist:
            export_image_to_fits(pb, "%s/test_gridding_aterm_pb.fits" % self.dir)
        gcf, cf = create_awterm_convolutionfunction(self.model, make_pb=make_pb, nw=1, oversampling=1, support=16,
                                                    use_aaf=True, polarisation_frame=self.vis_pol)
        griddata = create_griddata_from_image(self.model, polarisation_frame=self.vis_pol)
        griddata, sumwt = grid_blockvisibility_to_griddata(self.vis, griddata=griddata, cf=cf)
        cim = fft_griddata_to_image(griddata, gcf, gcf)
        cim = normalize_sumwt(cim, sumwt)
        im = convert_polimage_to_stokes(cim)
        if self.persist:
            export_image_to_fits(im, '%s/test_gridding_dirty_aterm_noover.fits' % self.dir)
        self.check_peaks(im, 97.10594988489598)
    
    def test_griddata_invert_box(self):
        self.actualSetUp(zerow=True)
        gcf, cf = create_box_convolutionfunction(self.model, polarisation_frame=self.vis_pol)
        griddata = create_griddata_from_image(self.model, polarisation_frame=self.vis_pol)
        griddata, sumwt = grid_blockvisibility_to_griddata(self.vis, griddata=griddata, cf=cf)
        cim = fft_griddata_to_image(griddata, gcf, gcf)
        cim = normalize_sumwt(cim, sumwt)
        im = convert_polimage_to_stokes(cim)
        if self.persist:
            export_image_to_fits(im, '%s/test_gridding_dirty_box.fits' % self.dir)
        self.check_peaks(im, 97.10594988489598, tol=1e-7)
    
    def check_peaks(self, im, peak, tol=1e-6):
        assert numpy.abs(im["pixels"].data[self.peak] - peak) < tol, im["pixels"].data[self.peak]
    
    def test_griddata_invert_wterm(self):
        self.actualSetUp(zerow=False)
        gcf, cf = create_awterm_convolutionfunction(self.model, nw=100, wstep=8.0, oversampling=4, support=32,
                                                    use_aaf=True, polarisation_frame=self.vis_pol)
        
        griddata = create_griddata_from_image(self.model, polarisation_frame=self.vis_pol)
        griddata, sumwt = grid_blockvisibility_to_griddata(self.vis, griddata=griddata, cf=cf)
        cim = fft_griddata_to_image(griddata, gcf, gcf)
        cim = normalize_sumwt(cim, sumwt)
        im = convert_polimage_to_stokes(cim)
        if self.persist:
            export_image_to_fits(im, '%s/test_gridding_dirty_wterm.fits' % self.dir)
        self.check_peaks(im, 97.13206509100314)
    
    def test_griddata_invert_wterm_noover(self):
        self.actualSetUp(zerow=False)
        gcf, cf = create_awterm_convolutionfunction(self.model, nw=100, wstep=8.0, oversampling=1, support=32,
                                                    use_aaf=True, polarisation_frame=self.vis_pol)
        
        # cf_image = convert_convolutionfunction_to_image(cf)
        # cf_image.data = numpy.real(cf_image.data)
        # if self.persist:
        #     export_image_to_fits(cf_image, "%s/test_gridding_wterm_cf.fits" % self.dir)
        
        griddata = create_griddata_from_image(self.model, polarisation_frame=self.vis_pol)
        griddata, sumwt = grid_blockvisibility_to_griddata(self.vis, griddata=griddata, cf=cf)
        cim = fft_griddata_to_image(griddata, gcf, gcf)
        cim = normalize_sumwt(cim, sumwt)
        im = convert_polimage_to_stokes(cim)
        if self.persist:
            export_image_to_fits(im, '%s/test_gridding_dirty_wterm.fits' % self.dir)
        self.check_peaks(im, 97.1343833)
    
    def test_griddata_check_cf_grid_wcs(self):
        self.actualSetUp(zerow=False)
        gcf, cf = create_awterm_convolutionfunction(self.model, nw=11, wstep=80.0, oversampling=4, support=32,
                                                    use_aaf=True, polarisation_frame=self.vis_pol)
        griddata = create_griddata_from_image(self.model, polarisation_frame=self.vis_pol)
        assert cf.convolutionfunction_acc.cf_wcs.wcs.cdelt[0] == griddata.griddata_acc.griddata_wcs.wcs.cdelt[0], \
            str(cf.convolutionfunction_acc.cf_wcs.wcs.cdelt[:2]) + str(griddata.griddata_acc.griddata_wcs.wcs.cdelt[:2])
        assert cf.convolutionfunction_acc.cf_wcs.wcs.cdelt[1] == griddata.griddata_acc.griddata_wcs.wcs.cdelt[1], \
            str(cf.convolutionfunction_acc.cf_wcs.wcs.cdelt[:2]) + str(griddata.griddata_acc.griddata_wcs.wcs.cdel[:2])
    
    @unittest.skip("Not reliable")
    def test_griddata_predict_aterm(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIQUV"))
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0, use_local=False)
        modelIQUV = convert_stokes_to_polimage(self.model, self.vis.blockvisibility_acc.polarisation_frame)
        griddata = create_griddata_from_image(modelIQUV, polarisation_frame=self.vis_pol)
        gcf, cf = create_awterm_convolutionfunction(modelIQUV, make_pb=make_pb, nw=1,
                                                    oversampling=16, support=32,
                                                    use_aaf=True, polarisation_frame=self.vis_pol)
        griddata = fft_image_to_griddata(modelIQUV, griddata, gcf)
        newvis = degrid_blockvisibility_from_griddata(self.vis, griddata=griddata, cf=cf)
        qa = qa_visibility(newvis)
        assert qa.data['rms'] < 160.0, str(qa)
    
    def test_griddata_predict_wterm(self):
        self.actualSetUp(zerow=False, image_pol=PolarisationFrame("stokesIQUV"))
        gcf, cf = create_awterm_convolutionfunction(self.model, nw=11, wstep=80.0, oversampling=4, support=32,
                                                    use_aaf=True, polarisation_frame=self.vis_pol)
        modelIQUV = convert_stokes_to_polimage(self.model, self.vis.blockvisibility_acc.polarisation_frame)
        griddata = create_griddata_from_image(modelIQUV, polarisation_frame=self.vis_pol)
        griddata = fft_image_to_griddata(modelIQUV, griddata, gcf)
        newvis = degrid_blockvisibility_from_griddata(self.vis, griddata=griddata, cf=cf)
        newvis['vis'].data[...] -= self.vis['vis'].data[...]
        qa = qa_visibility(newvis)
        self.plot_vis(newvis, 'wterm')
        assert qa.data['rms'] < 27.0, str(qa)
    
    def test_griddata_predict_awterm(self):
        self.actualSetUp(zerow=False, image_pol=PolarisationFrame("stokesIQUV"))
        modelIQUV = convert_stokes_to_polimage(self.model, self.vis.blockvisibility_acc.polarisation_frame)
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0, use_local=False)
        pb = make_pb(modelIQUV)
        if self.persist:
            export_image_to_fits(pb, "%s/test_gridding_awterm_pb.fits" % self.dir)
        gcf, cf = create_awterm_convolutionfunction(self.model, make_pb=make_pb, nw=100, wstep=8.0,
                                                    oversampling=4, support=32, use_aaf=True,
                                                    polarisation_frame=self.vis_pol)
        griddata = create_griddata_from_image(modelIQUV, polarisation_frame=self.vis_pol)
        griddata = fft_image_to_griddata(modelIQUV, griddata, gcf)
        newvis = degrid_blockvisibility_from_griddata(self.vis, griddata=griddata, cf=cf)
        qa = qa_visibility(newvis)
        assert qa.data['rms'] < 160.5, str(qa)
        self.plot_vis(newvis, 'awterm')
    
    def test_griddata_blockvisibility_weight(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIQUV"))
        gcf, cf = create_pswf_convolutionfunction(self.model, polarisation_frame=self.vis_pol)
        gd = create_griddata_from_image(self.model, polarisation_frame=self.vis_pol)
        gd_list = [grid_blockvisibility_weight_to_griddata(self.vis, gd) for i in range(10)]
        assert numpy.max(numpy.abs(gd_list[0][0]["pixels"].data)) > 10.0
        gd, sumwt = griddata_merge_weights(gd_list)
        self.vis = griddata_blockvisibility_reweight(self.vis, gd)
        gd, sumwt = grid_blockvisibility_to_griddata(self.vis, griddata=gd, cf=cf)
        cim = fft_griddata_to_image(gd, gcf, gcf)
        cim = normalize_sumwt(cim, sumwt)
        im = convert_polimage_to_stokes(cim)
        if self.persist:
            export_image_to_fits(im, '%s/test_gridding_dirty_2d_uniform_block.fits' % self.dir)
        self.check_peaks(im, 99.40822097)
    
    def test_griddata_blockvisibility_weight_I(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesI"))
        gcf, cf = create_pswf_convolutionfunction(self.model, polarisation_frame=self.vis_pol)
        gd = create_griddata_from_image(self.model, polarisation_frame=self.vis_pol)
        gd_list = [grid_blockvisibility_weight_to_griddata(self.vis, gd) for i in range(10)]
        assert numpy.max(numpy.abs(gd_list[0][0]["pixels"].data)) > 10.0
        gd, sumwt = griddata_merge_weights(gd_list)
        self.vis = griddata_blockvisibility_reweight(self.vis, gd)
        gd, sumwt = grid_blockvisibility_to_griddata(self.vis, griddata=gd, cf=cf)
        cim = fft_griddata_to_image(gd, gcf, gcf)
        cim = normalize_sumwt(cim, sumwt)
        im = convert_polimage_to_stokes(cim)
        if self.persist:
            export_image_to_fits(im, '%s/test_gridding_dirty_2d_IQ_uniform_block.fits' % self.dir)
        self.check_peaks(im, 99.40822097)
    
    def test_griddata_blockvisibility_weight_IQ(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIQ"))
        gcf, cf = create_pswf_convolutionfunction(self.model, polarisation_frame=self.vis_pol)
        gd = create_griddata_from_image(self.model, polarisation_frame=self.vis_pol)
        gd_list = [grid_blockvisibility_weight_to_griddata(self.vis, gd) for i in range(10)]
        assert numpy.max(numpy.abs(gd_list[0][0]["pixels"].data)) > 10.0
        gd, sumwt = griddata_merge_weights(gd_list)
        self.vis = griddata_blockvisibility_reweight(self.vis, gd)
        gd, sumwt = grid_blockvisibility_to_griddata(self.vis, griddata=gd, cf=cf)
        cim = fft_griddata_to_image(gd, gcf, gcf)
        cim = normalize_sumwt(cim, sumwt)
        im = convert_polimage_to_stokes(cim)
        if self.persist:
            export_image_to_fits(im, '%s/test_gridding_dirty_2d_IQ_uniform_block.fits' % self.dir)
        self.check_peaks(im, 99.40822097)
    
    def plot_vis(self, newvis, title=''):
        if self.doplot:
            import matplotlib.pyplot as plt
            r = numpy.sqrt(newvis.u ** 2 + newvis.v ** 2)
            for pol in range(4):
                plt.plot(newvis.w, numpy.real(newvis.vis[:, pol]), '.')
            plt.xlim(150, 300)
            plt.title('Prediction error for %s gridding' % title)
            plt.xlabel('W (wavelengths)')
            plt.ylabel('Real part of visibility prediction error')
            plt.show(block=False)


if __name__ == '__main__':
    unittest.main()
