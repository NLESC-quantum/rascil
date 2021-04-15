import numpy as np
import pytest

from scipy.signal.windows import gaussian as scipy_gaussian

from rascil.apps.ci_checker.ci_diagnostics import (
    qa_image_bdsf,
    plot_name,
    gaussian,
    histogram,
    plot_with_running_mean,
    source_region_mask,
    _radial_profile,
    power_spectrum,
)
from rascil.data_models import rascil_path


class MockGaussianObject:
    def __init__(self):
        self.centre_pix = [60.0, 180.0]


class MockBDSFImage:
    """
    Mock class representing a BDSF image, only with methods
    and properties used in the tested functions
    """

    def __init__(self):
        # see bdsf.readimage.Op_readimage.__call__
        # 4D array: (nstokes, nchannels, imag_size_x, image_size_y)
        self.image_arr = np.ones((1, 1, 10, 10))

        # list of objects of type bdsf.gausfit.Gaussian
        # couldn't decipher what units the values of .centre_pix
        #   (the only method used in our code) are
        # it is also not clear how many gaussians an image will have
        self.gaussians = [
            MockGaussianObject(),
            MockGaussianObject(),
            MockGaussianObject(),
        ]

    def pixel_beam(self):
        """
        beam = [major beam, minor beam, position angle]
        see: bdsf.readimage.Op_readimage.init_beam and
             bdsf.readimage.Op_readimage.init_beam.pixel_beam
        """
        return [0.5, 0.5, 0.0]


def test_qa_image_bdsf():
    im_data = np.array(
        [
            [1.0, 1.0, 1.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, -3.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )

    result = qa_image_bdsf(im_data)

    assert result["shape"] == "(5, 5)"
    assert result["max"] == 2.0
    assert result["min"] == -3.0
    assert result["maxabs"] == 3.0
    assert result["sum"] == 22.0
    assert result["medianabsdevmedian"] == 0.0


@pytest.mark.parametrize(
    "image_name, expected",
    [
        ("my_img.fits", "my_img_coloured_line"),
        ("myfig", "myfig_coloured_line"),
        ("my_fig.bla", "my_fig.bla_coloured_line"),  # extensions is not recognized
    ],
)
def test_plot_name(image_name, expected):
    """TODO: do we want to account for other extensions? is current behaviour acceptable?"""
    image_type = "coloured"
    plot_type = "line"

    result = plot_name(image_name, image_type, plot_type)

    assert result == expected


def test_gaussian():
    """
    Testing the gaussian function against the
    scipy gaussian window function, with amplitude of 1.
    """

    # scipy_gaussian defines the data array from the input length of the array as follows
    array_length = 9
    data = np.arange(0, array_length) - (array_length - 1.0) / 2.0
    # the resulting data array is:
    #   data = [-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.]

    mean = np.mean(data)
    std = np.std(data)

    expected_gaussian = scipy_gaussian(array_length, std)

    result = gaussian(data, 1.0, mean, std)
    assert (result == expected_gaussian).all()


def test_source_region_mask():
    my_image = MockBDSFImage()

    result = source_region_mask(my_image)
    sourced_mask = result[0]
    background_mask = result[1]

    # TODO:
    # what are the above when running
    #   tests.apps.ci_checker.test_ci_checker_main.test_continuum_imaging_checker
    #
    # test1: input image == source_mask.data == background_mask.data
    # test2: same as for test1
    # test3: same
    # test4: same
    # test5: same
    #
    # it also looks to me that for all tests the source_mask.mask is all True
    #   and background_mask.mask is all False
    # is this the expected behaviour?
    # is there a bug / conversion problem when the gaussian.centre_pix values are used?

    print("Done")


def test_radial_profile():
    """
    with a 5x5 array of 1s, with the first element being a 2 instead of 1,
    centre of image is left at default: (2, 2)

    the following arrays are the image_ravel_array (np.ravel(img),
    and the radius_ravel_array (np.ravel(r)) (np.ravel flattens multiD arrays into 1D):
        img_ravel = np.array([2]+[1] * 24)
        radius_ravel = np.array(
            [2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 1, 0, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2]
        )
    then the two calls of np.bincount() will return:
        weighted_bincount = np.array([1., 8., 17.])
            --> 0 appears once, 1 appears eight times, 2 appears seventeen times in the radius_ravel
                array weighted by the img_ravel array (the first 2 in radius_ravel shall be counted
                twice, because the first element of the img_ravel array is two)
        unweighted_bincount = np.array([1., 8., 16.])
            --> when radius_ravel is not weighted with img_ravel, the first 2 is only counted once,
            hence the unweighted array contains a 16 at position two, not a 17.

    _radial_profile functions returns np.array([1., 1., 17./16.])
    """
    img = np.ones((5, 5))
    img[0][0] = 2
    result = _radial_profile(img)

    assert (result == np.array([1., 1., 17./16.])).all()


def test_radial_profile_custom_centre():
    """
    same as test_radial_profile, but here we provide the centre of the image as an input

    centre = (2, 3)
    img.ravel() = np.array([2]+[1] * 24)
    radius.ravel() = np.array(
        [3, 2, 2, 2, 2, 3, 2, 1, 1, 1, 3, 2, 1, 0, 1, 3, 2, 1, 1, 1, 3, 2, 2, 2, 2]
    )

    weighted_bincount = np.array([1., 8., 11., 6.])
        --> weight of 2 at position zero, where a 3 is in the radius.ravel() array -->
        --> the 3 has to be counted twice
    unweighted_bincount = np.array([1., 8., 11., 5.])
    """
    img = np.ones((5, 5))
    img[0][0] = 2
    centre = (2, 3)
    result = _radial_profile(img, centre)

    assert (result == np.array([1., 1., 1., 6./5.])).all()


def test_power_spectrum():
    """
    TODO: do we need units for the log plot of the power spectrum axes?
        what is K?
    """
    img_file = rascil_path(
        "test_results/test-imaging-pipeline-dask_continuum_imaging_residual.fits"
    )

    result = power_spectrum(img_file, 5.0e-4)

    print("Done")
