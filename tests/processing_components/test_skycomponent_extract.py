""" Unit tests

"""

import logging
import pytest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import create_low_test_skymodel_from_gleam, \
    extract_skycomponents_from_image
from rascil.processing_components import find_nearest_skycomponent

log = logging.getLogger('rascil-logger')

log.setLevel(logging.INFO)


@pytest.mark.parametrize("cellsize, npixel, component_extraction", [
#    (0.001, 512, "pixels"),
    (0.005, 1024, "pixels")
])
def test_skycomponent_extract(npixel, cellsize, component_extraction):
    dec = -40.0 * u.deg
    cellsize = 0.001
    
    frequency = numpy.linspace(0.9e8, 1.1e8, 20)
    print(frequency)
    channel_bandwidth = numpy.array(len(frequency) * [frequency[1]-frequency[0]])
    phasecentre = SkyCoord(ra=+30.0 * u.deg, dec=dec, frame='icrs', equinox='J2000')
    
    # Create skymodel with components for sources > 1.0Jy and images for sources < 1.0Jy
    sm = create_low_test_skymodel_from_gleam(npixel=512, cellsize=cellsize,
                                             polarisation_frame=PolarisationFrame("stokesI"),
                                             frequency=frequency, channel_bandwidth=channel_bandwidth,
                                             phasecentre=phasecentre,
                                             flux_limit=0.3,
                                             flux_max=10.0,
                                             flux_threshold=1.0)
    assert numpy.max(numpy.abs(sm.image["pixels"].data)) > 0.0, "No flux in image"
    
    # Create skymodel with all components> 1.0Jy
    all_sm = create_low_test_skymodel_from_gleam(npixel=512, cellsize=cellsize,
                                                 polarisation_frame=PolarisationFrame("stokesI"),
                                                 frequency=frequency, channel_bandwidth=channel_bandwidth,
                                                 phasecentre=phasecentre,
                                                 flux_limit=1.0,
                                                 flux_max=10.0,
                                                 flux_threshold=1.0)
    # Now extract all sources > 1.0Jy
    newim, newsc = extract_skycomponents_from_image(sm.image, component_threshold=1.0,
                                                    component_extraction=component_extraction)
    
    for i, sc in enumerate(newsc):
        log.info(f"{i}, {sc}")
    for i, sc in enumerate(newsc):
        fsc, sep = find_nearest_skycomponent(sc.direction, all_sm.components)
        assert sep < cellsize, "Separation {sep} exceeds cellsize {cellsize}"
