""" Unit tests for visibility weighting
"""
import logging
import os
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import create_image_from_visibility
from rascil.processing_components import export_image_to_fits
from rascil.processing_components import invert_2d
from rascil.processing_components import (
    weight_visibility,
    taper_visibility_gaussian,
    taper_visibility_tukey,
    fit_psf,
)
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility.base import create_blockvisibility

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)


class TestWeighting(unittest.TestCase):
    def setUp(self):
        from rascil.data_models.parameters import rascil_path
        
        self.dir = rascil_path("test_results")
        self.npixel = 512
        
        self.persist = os.getenv("RASCIL_PERSIST", False)
    
    def actualSetUp(
            self, time=None, dospectral=False, image_pol=PolarisationFrame("stokesI")
    ):
        self.lowcore = create_named_configuration("LOWBD2", rmax=600)
        self.times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 5)
        
        if time is not None:
            self.times = time
        log.info("Times are %s" % self.times)
        
        if dospectral:
            self.nchan = 3
            self.frequency = numpy.array([0.9e8, 1e8, 1.1e8])
            self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7])
        else:
            self.frequency = numpy.array([1e8])
            self.channel_bandwidth = numpy.array([1e7])
        
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
        
        if dospectral:
            numpy.array([f, 0.8 * f, 0.6 * f])
        else:
            numpy.array([f])
        
        self.phasecentre = SkyCoord(
            ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.componentvis = create_blockvisibility(
            self.lowcore,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            phasecentre=self.phasecentre,
            weight=1.0,
            polarisation_frame=self.vis_pol,
        )
        
        self.uvw = self.componentvis["uvw"].data
        self.componentvis["vis"].data *= 0.0
        
        # Create model
        self.model = create_image_from_visibility(
            self.componentvis,
            npixel=self.npixel,
            cellsize=0.0005,
            nchan=len(self.frequency),
            polarisation_frame=self.image_pol,
        )
    
    def test_tapering_Gaussian(self):
        """Apply a Gaussian taper to the visibility and check to see if the PSF size is close"""
        self.actualSetUp()
        size_required = 0.020
        self.componentvis = weight_visibility(
            self.componentvis, self.model, algoritm="uniform"
        )
        self.componentvis = taper_visibility_gaussian(
            self.componentvis, beam=size_required
        )
        psf, sumwt = invert_2d(self.componentvis, self.model, dopsf=True)
        if self.persist:
            export_image_to_fits(
                psf,
                "%s/test_weighting_gaussian_taper_psf.fits" % (self.dir),
            )
        fit = fit_psf(psf)
        
        assert (
                numpy.abs(fit["bmaj"] - 1.279952050682638) < 1
        ), "Fit should be %f, actually is %f" % (1.279952050682638, fit["bmaj"])
    
    def test_tapering_tukey(self):
        """Apply a Tukey window taper and output the psf and FT of the PSF. No quantitative check.

        :return:
        """
        self.actualSetUp()
        self.componentvis = weight_visibility(
            self.componentvis, self.model, algorithm="uniform"
        )
        self.componentvis = taper_visibility_tukey(self.componentvis, tukey=0.1)
        psf, sumwt = invert_2d(self.componentvis, self.model, dopsf=True)
        if self.persist:
            export_image_to_fits(
                psf,
                "%s/test_weighting_tukey_taper_psf.fits" % (self.dir),
            )
        fit = fit_psf(psf)
        assert (
                numpy.abs(fit["bmaj"] - 0.14492670913355402) < 1.0
        ), "Fit should be %f, actually is %f" % (0.14492670913355402, fit["bmaj"])


if __name__ == "__main__":
    unittest.main()
