""" Imaging the MeasurementSet sim-2.ms

"""
import logging
import sys

import numpy

from rascil.data_models import PolarisationFrame, rascil_data_path
from rascil.processing_components import create_blockvisibility_from_ms, \
    export_image_to_fits, qa_image, \
    deconvolve_cube, restore_cube, create_image_from_visibility, invert_2d
from rascil.workflows import invert_list_serial_workflow

log = logging.getLogger('logger')

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))

if __name__ == '__main__':
    results_dir = './'

    vt = create_blockvisibility_from_ms(rascil_data_path('vis/sim-2.ms'), start_chan=35,
                                         end_chan=39)[0]
    vt.configuration.diameter[...] = 35.0
    
    a2r = numpy.pi / (180.0 * 3600.0)
    model = create_image_from_visibility(vt, cellsize=20.0 * a2r, npixel=512,
                                         polarisation_frame=PolarisationFrame('stokesIQUV'))
    dirty, sumwt = invert_2d(vt, model)
    psf, sumwt = invert_2d(vt, model, dopsf=True)
    export_image_to_fits(dirty, '%s/rascil_imaging_sim_2_dirty.fits' % (results_dir))
    export_image_to_fits(psf, '%s/rascil_imaging_sim_2_psf.fits' % (results_dir))

    # Deconvolve using msclean
    comp, residual = deconvolve_cube(dirty, psf, niter=10000, threshold=0.001,
                                     fractional_threshold=0.001,
                                     algorithm='msclean',
                                     window_shape='quarter', gain=0.7,
                                     scales=[0, 3, 10, 30])

    restored = restore_cube(comp, psf, residual)

    print(qa_image(restored))
    export_image_to_fits(restored,
                         '%s/rascil_imaging_sim_2_restored.fits' % (results_dir))
    export_image_to_fits(residual,
                         '%s/rascil_imaging_sim_2_residual.fits' % (results_dir))
