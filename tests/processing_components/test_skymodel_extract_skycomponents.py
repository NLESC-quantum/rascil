""" Unit tests for skymodel

"""

import logging

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import (
    create_low_test_skymodel_from_gleam,
    extract_skycomponents_from_skymodel,
)
from rascil.processing_components import find_nearest_skycomponent

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)


def test_skymodel_extract_skycomponents():
    dec = -40.0 * u.deg
    cellsize = 0.001

    frequency = numpy.array([1e8])
    channel_bandwidth = numpy.array([1e6])
    phasecentre = SkyCoord(ra=+30.0 * u.deg, dec=dec, frame="icrs", equinox="J2000")

    # :param flux_limit: Weakest component
    # :param flux_max: Maximum strength component to be included in components
    # :param flux_threshold: Split between components (brighter) and image (weaker)

    # Create skymodel with only components for sources > 0.3Jy
    sm_all_components = create_low_test_skymodel_from_gleam(
        npixel=512,
        cellsize=cellsize,
        polarisation_frame=PolarisationFrame("stokesI"),
        frequency=frequency,
        channel_bandwidth=channel_bandwidth,
        phasecentre=phasecentre,
        flux_limit=3.0,
        flux_max=10.0,
        flux_threshold=3.0,
        applybeam=False,
    )

    # Create skymodel with all components in the image
    sm_all_image = create_low_test_skymodel_from_gleam(
        npixel=512,
        cellsize=cellsize,
        polarisation_frame=PolarisationFrame("stokesI"),
        frequency=frequency,
        channel_bandwidth=channel_bandwidth,
        phasecentre=phasecentre,
        flux_limit=3.0,
        flux_max=10.0,
        flux_threshold=10.0,
        applybeam=False,
    )

    # Now extract all sources > 1.0Jy
    sm_found_components = extract_skycomponents_from_skymodel(
        sm_all_image, component_threshold=0.3, component_method="fit"
    )
    assert len(sm_found_components.components) > 0, "No components found"

    for i, sc in enumerate(sm_found_components.components):
        fsc, sep = find_nearest_skycomponent(sc.direction, sm_all_components.components)
        assert sep < cellsize, "Separation {sep} exceeds cellsize {cellsize}"
