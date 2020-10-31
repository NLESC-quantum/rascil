""" Unit tests for pipelines expressed via dask.delayed


"""
import functools
import os
import sys
import unittest

from matplotlib import pyplot as plt

from rascil.data_models.polarisation import PolarisationFrame

from rascil.data_models.parameters import rascil_path, rascil_data_path

from rascil.processing_components import copy_image
from rascil.processing_components import create_blockvisibility_from_ms, show_image, create_pb, \
    create_image_from_visibility, invert_awprojection, create_awterm_convolutionfunction, \
    export_convolutionfunction_to_fits, export_image_to_fits, weight_visibility

import logging

log = logging.getLogger("rascil-logger")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

results_dir = rascil_path("test_results")

log = logging.getLogger('rascil-logger')

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestImaging2D(unittest.TestCase):
    def setUp(self):
        from rascil.data_models.parameters import rascil_path
        self.dir = rascil_path('test_results')
        
        self.persist = os.getenv("RASCIL_PERSIST", False)
    
    def test_mosaic(self):
        vis_list = create_blockvisibility_from_ms(rascil_data_path('vis/xcasa.ms'))
        
        cellsize = 0.00001
        
        mid_frequency = [0.5 * (vis_list[0].frequency.values[0] + vis_list[1].frequency.values[0])]
        total_bandwidth = [vis_list[0].channel_bandwidth.values[0] + vis_list[1].channel_bandwidth.values[0]]
        model = create_image_from_visibility(vis_list[0], cellsize=cellsize, npixel=512, nchan=1,
                                             frequency=mid_frequency, channel_bandwidth=total_bandwidth,
                                             imagecentre=vis_list[0].phasecentre,
                                             polarisation_frame=PolarisationFrame('stokesIQUV'))
        make_pb = functools.partial(create_pb, telescope='VLA', pointingcentre=vis_list[0].phasecentre)
        gcfcf = create_awterm_convolutionfunction(model, make_pb=make_pb,
                                                  polarisation_frame=PolarisationFrame('circular'),
                                                  oversampling=17, support=10)
        mosaic = copy_image(model)
        
        for vt in vis_list:
            vt = weight_visibility(vt, model)
            dirty, sumwt = invert_awprojection(vt, model, gcfcf=gcfcf)
            mosaic.data.values += dirty.data.values
        
        show_image(mosaic, cm='Greys', title='Linear mosaic')
        plt.show()
        
        if self.persist:
            export_image_to_fits(mosaic, "{}/test_mosaic_dirty.fits".format(results_dir))
            export_convolutionfunction_to_fits(gcfcf[1], "{}/test_mosaic_cf.fits".format(results_dir))


if __name__ == '__main__':
    unittest.main()
