import os
import numpy
import pytest
import logging

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
from rascil.processing_components.skycomponent import fit_skycomponent_spectral_index
from rascil.workflows.rsexecute.execution_support import rsexecute

log = logging.getLogger("rascil-logger")
log.setLevel(logging.WARNING)

USE_DASK = False


def write_to_txt(filename, components):

    f = open(filename, "w")
    f.write(
        "# RA(deg), Dec(deg), I (Jy), Q (Jy), U (Jy), V (Jy), Ref. freq. (Hz), Spectral index\n"
    )

    for cmp in components:
        coord_ra = cmp.direction.ra.degree
        coord_dec = cmp.direction.dec.degree
        spec_indx = fit_skycomponent_spectral_index(cmp)
        nchan = len(cmp.frequency)
        central_freq = cmp.frequency[nchan // 2]
        f.write(
            "%.6f, %.6f, %10.6e, %10.6e, %10.6e, %10.6e, %10.6e, %10.6e \n"
            % (
                coord_ra,
                coord_dec,
                cmp.flux[nchan // 2],
                0.0,
                0.0,
                0.0,
                central_freq,
                spec_indx,
            )
        )
    f.close()

    return


@pytest.fixture(scope="module")
def set_up_dask():
    rsexecute.set_client(use_dask=USE_DASK)
    yield
    rsexecute.close()


@pytest.fixture(scope="module")
def sky_comp_file():
    """
    Create a temporary HDF file with three SkyComponents.
    """
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

    skycomp_txt = rascil_path("test_results/test_generate_skymodel_list_skycomps.txt")
    write_to_txt(skycomp_txt, sky_com_list)

    yield skycomp_file, skycomp_txt

    os.remove(skycomp_file)
    os.remove(skycomp_txt)


@pytest.fixture(scope="module")
def mock_image():
    """
    Create a mock image for testing.
    """
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


def test_generate_skymodel_list(mock_image, set_up_dask):
    """
    Function correctly generates a list of SkyModels when there isn't
    an inout component file, using a source at the phase centre.
    """
    image = mock_image[0]
    phase_centre = mock_image[1]

    result = generate_skymodel_list([image])
    result = rsexecute.compute(result, sync=True)

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


def test_generate_skymodel_list_from_hdf_one_comp(
    mock_image, sky_comp_file, set_up_dask
):
    """
    Test that the code correctly chooses the brightest component read
    from an HDF file, when n_bright_sources=1
    """
    result = generate_skymodel_list(
        [mock_image[0]], input_file=sky_comp_file[0], n_bright_sources=1
    )
    result = rsexecute.compute(result, sync=True)

    assert len(result) == 1
    assert len(result[0].components) == 1
    assert (result[0].components[0].flux == numpy.array([[1.1], [2.2], [2.5]])).all()


def test_generate_skymodel_list_from_hdf_two_comps(
    mock_image, sky_comp_file, set_up_dask
):
    """
    Test that the code correctly chooses the n brightest components read
    from an HDF file, when n_bright_sources=n (n>1)
    """
    result = generate_skymodel_list(
        [mock_image[0]], input_file=sky_comp_file[0], n_bright_sources=2
    )
    result = rsexecute.compute(result, sync=True)

    assert len(result) == 1
    assert len(result[0].components) == 2
    assert (result[0].components[0].flux == numpy.array([[1.1], [2.2], [2.5]])).all()
    assert (result[0].components[1].flux == numpy.array([[1.0], [2.0], [2.5]])).all()


def test_generate_skymodel_list_from_hdf_all_comps(
    mock_image, sky_comp_file, set_up_dask
):
    """
    Test that the code uses all of the components read
    from an HDF file, when n_bright_sources=None and there is an input file.
    """
    result = generate_skymodel_list(
        [mock_image[0]], input_file=sky_comp_file[0], n_bright_sources=None
    )
    result = rsexecute.compute(result, sync=True)

    assert len(result) == 1
    assert len(result[0].components) == 3
    # The flux won't be sorted
    assert (result[0].components[1].flux == numpy.array([[1.1], [2.2], [2.5]])).all()
    assert (result[0].components[0].flux == numpy.array([[1.0], [2.0], [2.5]])).all()
    assert (result[0].components[2].flux == numpy.array([[0.4], [1.1], [1.6]])).all()


def test_generate_skymodel_list_from_txt_one_comp(
    mock_image, sky_comp_file, set_up_dask
):
    """
    Test that the code correctly chooses the brightest component read
    from a TXT file, when n_bright_sources=1

    Frequency of components are scaled to image frequency.
    """
    result = generate_skymodel_list(
        [mock_image[0]], input_file=sky_comp_file[1], n_bright_sources=1
    )
    result = rsexecute.compute(result, sync=True)

    expected = numpy.array([[1.34210651], [1.35848315], [1.37489533]])
    assert len(result) == 1
    assert len(result[0].components) == 1
    assert_almost_equal(result[0].components[0].flux, expected)


def test_generate_skymodel_list_from_txt_two_comps(
    mock_image, sky_comp_file, set_up_dask
):
    """
    Test that the code correctly chooses the n brightest components read
    from a TXT file, when n_bright_sources=n (n>1)

    Frequency of components are scaled to image frequency.
    """
    result = generate_skymodel_list(
        [mock_image[0]], input_file=sky_comp_file[1], n_bright_sources=2
    )
    result = rsexecute.compute(result, sync=True)

    expected1 = numpy.array([[1.34210651], [1.35848315], [1.37489533]])
    expected2 = numpy.array([[1.15816502], [1.1737968], [1.18948243]])

    assert len(result) == 1
    assert len(result[0].components) == 2
    assert_almost_equal(result[0].components[0].flux, expected1)
    assert_almost_equal(result[0].components[1].flux, expected2)


def test_generate_skymodel_list_from_txt_all_comps(
    mock_image, sky_comp_file, set_up_dask
):
    """
    Test that the code code uses all of the components read
    from a TXT file, when n_bright_sources=None

    Frequency of components are scaled to image frequency.
    """
    result = generate_skymodel_list(
        [mock_image[0]],
        input_file=sky_comp_file[1],
        n_bright_sources=None,
    )
    result = rsexecute.compute(result, sync=True)

    expected1 = numpy.array([[1.34210651], [1.35848315], [1.37489533]])
    expected2 = numpy.array([[1.15816502], [1.1737968], [1.18948243]])
    expected3 = numpy.array([[0.48249189], [0.49234909], [0.50230758]])

    assert len(result) == 1
    assert len(result[0].components) == 3
    # The flux are not sorted
    assert_almost_equal(result[0].components[0].flux, expected2)
    assert_almost_equal(result[0].components[1].flux, expected1)
    assert_almost_equal(result[0].components[2].flux, expected3)
