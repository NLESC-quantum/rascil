""" Unit processing_components for rascil advise

"""
import logging
import unittest
import sys

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rascil.apps.ci_imaging_checker import cli_parser, analyze_image
from rascil.data_models.parameters import rascil_path
from rascil.data_models.polarisation import PolarisationFrame
from rascil.data_models.memory_data_models import Image, Skycomponent
from rascil.processing_components.simulation import create_mid_simulation_components, find_pb_width_null
from rascil.processing_components.skycomponent import insert_skycomponent, find_skycomponent_matches
from rascil.processing_components.image import create_image, export_image_to_fits, smooth_image

log = logging.getLogger('rascil-logger')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))

class TestContinuumImagingChecker(unittest.TestCase):

  def make_mid_test_fits(self):

      self.frequency = 1.e8
      self.phasecentre = SkyCoord(ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
      self.npixel = 512
      self.cellsize = 0.0004
      hwhm_deg, null_az_deg, null_el_deg = find_pb_width_null(pbtype="MID", frequency=numpy.array([self.frequency]))

      hwhm = hwhm_deg * numpy.pi / 180.0
      fov_deg = 8.0 * 1.36e9 / self.frequency
      pb_npixel = 256
      d2r = numpy.pi / 180.0
      pb_cellsize = d2r * fov_deg / pb_npixel
      pbradius = 1.5
      pbradius = pbradius * hwhm

      flux_limit = 1.0
      original_components = create_mid_simulation_components(self.phasecentre, numpy.array([self.frequency]), flux_limit,
                                                           pbradius, pb_npixel, pb_cellsize,
                                                           show=False, fov=10)                                             
    
      self.components = original_components[0]
      self.components = sorted(self.components, key=lambda comp: numpy.max(comp.direction.ra))      
      log.info("Original components:")
      log.info(" RA Dec Flux ")
      for comp in self.components:
          log.info("%s %s %.3f"%(comp.direction.ra, comp.direction.dec, comp.flux[0]))

      self.model = create_image(npixel=self.npixel,
                                   cellsize=self.cellsize,
                                   phasecentre=self.phasecentre,
                                   frequency=numpy.array([self.frequency]),
                                   polarisation_frame=PolarisationFrame("stokesI"))
      
      self.model = insert_skycomponent(self.model, self.components)
      self.model = smooth_image(self.model, width=3.0)

      export_image_to_fits(self.model, rascil_path('test_results/test_ci_checker.fits'))

  def setUp(self) -> None:

      self.dir = rascil_path('test_results')

      parser = cli_parser()
      self.args = parser.parse_args([])
      self.args.ingest_fitsname = rascil_path("test_results/test_ci_checker.fits")

      self.args.finder_beam_min = 0.5      
      #self.args.finder_beam_maj = 0.3

  def test_ci_checker(self):

      self.make_mid_test_fits()

      out = analyze_image(self.args)

      # check results
      out = sorted(out, key=lambda comp: numpy.max(comp.direction.ra)) 
      log.info("Identified components:")
      for comp in out:
          log.info("%s %s %.3f"%(comp.direction.ra, comp.direction.dec, comp.flux[0]))

      assert len(out) <= len(self.components)
      log.info("BDSF expected to find %d sources, but found %d sources"%(len(self.components), len(out)))
      matches = find_skycomponent_matches(out, self.components, tol=1e-3)
      log.info("Found matches as follows.")
      log.info("BDSF Original Separation")
      for match in matches:
          log.info("%d %d %10.6e" %(match[0], match[1], match[2])) 



if __name__ == '__main__':
    unittest.main()
