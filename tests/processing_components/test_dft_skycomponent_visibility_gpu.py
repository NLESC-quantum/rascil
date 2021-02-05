""" Unit tests for visibility operations


"""

import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord
from numpy.testing import assert_allclose

from rascil.data_models.memory_data_models import Skycomponent
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.imaging.dft import dft_skycomponent_visibility, idft_visibility_skycomponent
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility.base import create_blockvisibility, create_blockvisibility, \
    phaserotate_visibility

class TestVisibilityDFTOperationsGPU(unittest.TestCase):
    def setUp(self):
        pass
        
    def init(self, ntimes=2, nchan=10, ncomp=100):

        self.lowcore = create_named_configuration('LOW')
        self.times = (numpy.pi / 43200.0) * numpy.linspace(0.0, 300.0, ntimes)
        
        self.frequency = numpy.linspace(1.0e8, 1.1e8, nchan)
        self.channel_bandwidth = numpy.array(nchan * [1e7 / nchan])
        self.flux = numpy.array(nchan * [100.0, 20.0, -10.0, 1.0]).reshape([nchan, 4])

        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.compabsdirection = SkyCoord(ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        pcof = self.phasecentre.skyoffset_frame()
        self.compreldirection = self.compabsdirection.transform_to(pcof)

        self.comp = ncomp * [Skycomponent(direction=self.compreldirection, frequency=self.frequency,
                                          flux=self.flux)]

    # @unittest.skip("Don't run the slow version in CI")
    def test_dft_stokesiquv_blockvisibility(self):
        try:
            import cupy
            compute_kernels = ['gpu_cupy_einsum', 'cpu_einsum', 'cpu_numpy', 'cpu_unrolled']
        except ModuleNotFoundError:
            compute_kernels = ['cpu_einsum', 'cpu_numpy', 'cpu_unrolled']

        self.init(ntimes=2, nchan=10, ncomp=100)
        for compute_kernel in compute_kernels:
            import time
            start = time.time()
            self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                              channel_bandwidth=self.channel_bandwidth,
                                              phasecentre=self.phasecentre, weight=1.0,
                                              polarisation_frame=PolarisationFrame("linear"))
            self.vismodel = dft_skycomponent_visibility(self.vis, self.comp, compute_kernel=compute_kernel)
            print(f"{compute_kernel} {time.time() - start:.3}s")
            assert numpy.max(numpy.abs(self.vismodel["vis"].data)) > 0.0

    def test_dft_stokesiquv_blockvisibility_quick(self):
    
        self.init(ntimes=2, nchan=2, ncomp=2)
        for vpol in [PolarisationFrame("linear"), PolarisationFrame("circular")]:
            self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                              channel_bandwidth=self.channel_bandwidth,
                                              phasecentre=self.phasecentre, weight=1.0,
                                              polarisation_frame=vpol)
            self.vismodel = dft_skycomponent_visibility(self.vis, self.comp)
            assert numpy.max(numpy.abs(self.vismodel["vis"].data)) > 0.0


if __name__ == '__main__':
    unittest.main()
