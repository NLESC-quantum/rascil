""" Unit tests for skymodel

"""

import logging

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import create_low_test_skymodel_from_gleam, \
    extract_skycomponents_from_skymodel
from rascil.processing_components import find_nearest_skycomponent

log = logging.getLogger('rascil-logger')

log.setLevel(logging.WARNING)

def test_skymodel_update():
    dec = -40.0 * u.deg
    cellsize = 0.001
    
    frequency = numpy.array([1e8])
    channel_bandwidth = numpy.array([1e6])
    phasecentre = SkyCoord(ra=+30.0 * u.deg, dec=dec, frame='icrs', equinox='J2000')
    
    # Create skymodel with components for sources > 1.0Jy and images for sources < 1.0Jy
    sm = create_low_test_skymodel_from_gleam(npixel=512, cellsize=cellsize,
                                             polarisation_frame=PolarisationFrame("stokesI"),
                                             frequency=frequency, channel_bandwidth=channel_bandwidth,
                                             phasecentre=phasecentre,
                                             flux_limit=0.3,
                                             flux_max=1.0,
                                             flux_threshold=1.0)

    assert len(sm.components) == 0, "Components remaining in SkyModel"
    assert numpy.max(numpy.abs(sm.image["pixels"].data)) > 0.0, "No flux in image"
    
    # Create skymodel with all components> 1.0Jy
    all_sm = create_low_test_skymodel_from_gleam(npixel=512, cellsize=cellsize,
                                                 polarisation_frame=PolarisationFrame("stokesI"),
                                                 frequency=frequency, channel_bandwidth=channel_bandwidth,
                                                 phasecentre=phasecentre,
                                                 flux_limit=0.3,
                                                 flux_max=1.0,
                                                 flux_threshold=0.3)
    # Now extract all sources > 1.0Jy
    newsm = extract_skycomponents_from_skymodel(sm, component_threshold=0.5)
    
    for i, sc in enumerate(newsm.components):
        fsc, sep = find_nearest_skycomponent(sc.direction, all_sm.components)
        assert sep < cellsize, "Separation {sep} exceeds cellsize {cellsize}"
