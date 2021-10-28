import os
import numpy
import pytest
from numpy.testing import assert_almost_equal
from astropy.coordinates import SkyCoord
from astropy import units as u

from rascil.apps.rascil_imager import generate_skymodel_list
from rascil.data_models import (
    Skycomponent,
    export_skycomponent_to_hdf5,
    rascil_path,
    PolarisationFrame,
)
from rascil.processing_components.image.operations import create_image


@pytest.fixture(scope="module")
def skycomp_file():
    phase_centre = SkyCoord(
        ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
    )
    pol_frame = PolarisationFrame("stokesI")
    frequency = numpy.array([1.0e8, 1.5e8, 2.0e8])

    sky_com_list = [
        Skycomponent(
            direction=phase_centre,
            flux=numpy.array([[1.0], [2.0], [2.5]]),
            polarisation_frame=pol_frame,
            frequency=frequency,
        ),
        Skycomponent(
            direction=phase_centre,
            flux=numpy.array([[1.1], [2.2], [2.5]]),
            polarisation_frame=pol_frame,
            frequency=frequency,
        ),
        Skycomponent(
            direction=phase_centre,
            flux=numpy.array([[0.4], [1.1], [1.6]]),
            polarisation_frame=pol_frame,
            frequency=frequency,
        ),
    ]
    skycomp_file = rascil_path("test_results/test_generate_skymodel_list_skycomps.hdf5")
    export_skycomponent_to_hdf5(sky_com_list, skycomp_file)

    yield skycomp_file

    os.remove(skycomp_file)


@pytest.fixture(scope="module")
def mock_image():
    phase_centre = SkyCoord(
        ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
    )
    frequency = numpy.array([1.0e8, 1.5e8, 2.0e8])
    image = create_image(
        phasecentre=phase_centre,
        frequency=frequency,
        channel_bandwidth=numpy.array([1e6, 1e6, 1e6]),
    )

    return image, phase_centre


def test_generate_skymodel_list(mock_image):
    """
    Component and image frequencies will match because we create
    the component using information from the image
    """
    image = mock_image[0]
    phase_centre = mock_image[1]

    result = generate_skymodel_list([image])

    assert len(result) == 1
    assert len(result[0].components) == 1
    assert_almost_equal(
        result[0].components[0].direction.ra.value, phase_centre.ra.value
    )
    assert_almost_equal(
        result[0].components[0].direction.dec.value, phase_centre.dec.value
    )

    assert result[0].components[0].direction.ra.info.unit == phase_centre.ra.info.unit
    assert result[0].components[0].direction.dec.info.unit == phase_centre.dec.info.unit
    assert result[0].components[0].polarisation_frame.names == ["I"]


def test_generate_skymodel_list_from_hdf(mock_image, skycomp_file):
    """
    Note: frequency of image and components may not match!
    """
    result = generate_skymodel_list(
        [mock_image[0]], input_file=skycomp_file, n_bright_sources=1
    )

    assert len(result) == 1
    assert len(result[0].components) == 1
    assert (result[0].components[0].flux == numpy.array([[1.1], [2.2], [2.5]])).all()


def test_generate_skymodel_list_from_hdf_two_comps(mock_image, skycomp_file):
    """
    Note: frequency of image and components may not match!
    """
    result = generate_skymodel_list(
        [mock_image[0]], input_file=skycomp_file, n_bright_sources=2
    )

    assert len(result) == 1
    assert len(result[0].components) == 2
    assert (result[0].components[0].flux == numpy.array([[1.1], [2.2], [2.5]])).all()
    assert (result[0].components[1].flux == numpy.array([[1.0], [2.0], [2.5]])).all()


def test_generate_skymodel_list_from_hdf_all_comps(mock_image, skycomp_file):
    """
    Note: frequency of image and components may not match!
    """
    result = generate_skymodel_list(
        [mock_image[0]], input_file=skycomp_file, n_bright_sources=None
    )

    assert len(result) == 1
    assert len(result[0].components) == 3
    assert (result[0].components[0].flux == numpy.array([[1.1], [2.2], [2.5]])).all()
    assert (result[0].components[1].flux == numpy.array([[1.0], [2.0], [2.5]])).all()
    assert (result[0].components[2].flux == numpy.array([[0.4], [1.1], [1.6]])).all()
