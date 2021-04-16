from unittest.mock import patch, Mock, call, MagicMock

import numpy as np
import matplotlib.pyplot as plt
import pytest

from scipy.signal.windows import gaussian as scipy_gaussian

from rascil.apps.ci_checker.ci_diagnostics import (
    qa_image_bdsf,
    plot_name,
    gaussian,
    _get_histogram_data,
    histogram,
    plot_with_running_mean,
    source_region_mask,
    _radial_profile,
    _plot_power_spectrum,
    _save_power_spectrum_to_csv,
    power_spectrum,
    ci_checker_diagnostics,
)
from rascil.data_models import rascil_data_path

BASE_PATH = "rascil.apps.ci_checker.ci_diagnostics"


class MockGaussianObject:
    def __init__(self):
        self.centre_pix = [60.0, 180.0]


class MockBDSFImage:
    """
    Mock class representing a BDSF image, only with methods
    and properties used in the tested functions
    """

    def __init__(self):
        # not part of original PyBDSF image class
        self.gauss_mean = 1.0  # mean to be used in constructing self.resid_gaus_arr
        self.gauss_std = (
            0.5  # standard deviation to be used in constructing self.resid_gaus_arr
        )

        self.shape = (1, 1, 10, 10)

        # see bdsf.readimage.Op_readimage.__call__
        # 4D array: (nstokes, nchannels, imag_size_x, image_size_y)
        self.image_arr = np.ones(self.shape)

        self.raw_rms = np.sqrt(np.mean(self.image_arr ** 2))
        self.raw_mean = np.mean(self.image_arr)

        # see bdsf.make_residimage.Op_make_residimage
        # numpy array of image size --> shape: x_axis x y_axis
        self.resid_gaus_arr = self.create_gaussian_array()

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

    def create_gaussian_array(self):
        gaus_arr = np.random.normal(self.gauss_mean, self.gauss_std, self.shape[2:])
        return gaus_arr


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
        ("my_fig.bla", "my_fig.bla_coloured_line"),  # extension is not recognized
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


def test_get_histogram_data():
    _, ax = plt.subplots()
    my_image = MockBDSFImage()

    result = _get_histogram_data(my_image, ax)

    # the code creates 1000 bins; result[0] is the array of the centre of the bins
    assert len(result[0]) == 1000

    # fitted gaussian parameters
    fitted_params = result[1]

    # TODO: Are these reasonable boundaries for a fit?
    assert my_image.gauss_mean + 0.1 > fitted_params[1] > my_image.gauss_mean - 0.1
    assert my_image.gauss_std + 0.1 > abs(fitted_params[2]) > my_image.gauss_std - 0.1


@patch(BASE_PATH + ".plt")
@patch(
    BASE_PATH + "._get_histogram_data",
    Mock(
        return_value=(np.ones((10, 10)), [1.0, 0.5, 0.2])
    ),  # [1.0, 0.5, 0.2] --> amplitude, mean, std
)
def test_histogram(mock_plot):
    """
    Test that the various plt an ax calls are executed with the correct arguments.
    """
    # GIVEN
    mock_plot.return_value = Mock()
    mock_ax = Mock()
    mock_plot.subplots.return_value = (Mock(), mock_ax)

    # WHEN
    # the first arg doesn't have to be a real image, as the function that needs it is patched
    histogram("some-fake-img", "my_file.fits", "my_description")

    # THEN
    mock_ax.plot.assert_called_once()
    mock_ax.axvline.assert_called_once_with(
        0.5, color="C2", linestyle="--", label=f"mean: {0.5:.3e}", zorder=15
    )
    mock_ax.axvspan.assert_called_once_with(
        0.3, 0.7, facecolor="C2", alpha=0.3, zorder=10, label=f"stddev: {0.2:.3e}"
    )
    mock_ax.set_title.assert_called_once_with("my_description")
    mock_plot.savefig.assert_called_once_with("my_file_my_description_hist.png")


@pytest.mark.parametrize("description", ["my_description", "restored"])
@patch(BASE_PATH + ".plt")
def test_plot_with_running_mean(mock_plot, description):
    mock_plot.return_value = Mock()
    mock_fig = MagicMock()
    mock_plot.figure.return_value = mock_fig
    mock_fig.add_gridspec.return_value = np.zeros((4, 4))

    # two strings are added to test that a string can be anywhere within the stats dictionary
    # in a previous version, it only allowed for the 0th position
    stats = {
        "myString": "my string here",
        "mean": 0.5,
        "shape": "(5, 5)",
        "std": 0.2,
    }

    plot_with_running_mean(
        MockBDSFImage(),
        "my_image.fits",
        stats,
        "fake-projection",
        description=description,
    )

    assert mock_fig.add_subplot.call_count == 3
    assert mock_plot.text.call_count == 4  # three key-value pairs in stats dict
    mock_plot.savefig.assert_called_once_with(
        f"my_image_{description}_plot.png", pad_inches=-1
    )

    if description == "restored":
        assert (
            mock_plot.Circle.call_count == 3
        )  # MockBDSFImage.gaussians has 3 MockGaussianObject objects
        assert (
            mock_plot.Circle.call_args_list
            == [
                call(
                    (60.0, 180.0),
                    color="w",
                    fill=False,
                )
            ]
            * 3
        )

    else:
        mock_plot.Circle.assert_not_called()


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

    pass


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

    assert (result == np.array([1.0, 1.0, 17.0 / 16.0])).all()


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

    assert (result == np.array([1.0, 1.0, 1.0, 6.0 / 5.0])).all()


@patch(BASE_PATH + ".plt")
def test_plot_power_spectrum(mock_plot):
    mock_plot.return_value = Mock()
    profile = [1.0, 5.0, 17.0]
    theta_axis = [23.0, 44.0, 52.0]

    result = _plot_power_spectrum("my_image.fits", profile, theta_axis)
    expected_plot_name_string = "my_image_residual_power_spectrum"

    assert result == expected_plot_name_string
    assert mock_plot.gca.call_count == 6
    mock_plot.plot.assert_called_once_with(theta_axis, profile)
    mock_plot.savefig.assert_called_once_with(expected_plot_name_string + ".png")


def test_power_spectrum():
    """
    TODO: do we need units for the log plot of the power spectrum axes?
        what is K?
        what is profile and what is theta_axis?

    TODO: (fyi)
      Image() breaks without wcs and polarisation_frame specified, even though those are optional args
      rascil_image = Image(np.ones((1, 1, 5, 5)))
    """
    test_image = rascil_data_path("models/M31_canonical.model.fits")

    result = power_spectrum(test_image, 5.0e-4)

    expected_length = 182
    assert len(result[0]) == expected_length
    assert len(result[0]) == len(result[1])
    # is there anything else that can be reasonably tested/asserted?


@patch(BASE_PATH + ".SlicedLowLevelWCS", Mock())
@patch(BASE_PATH + ".source_region_mask")
@patch(BASE_PATH + ".qa_image_bdsf")
@patch(BASE_PATH + ".plot_with_running_mean")
@patch(BASE_PATH + ".histogram")
class TestCICheckerDiagnostics:
    """
    Test that the correct functions are called, and the correct number of times,
    depending on what "image_type" we run the ci_checker_diagnostics function with.

    In these tests, we mock the functions to check if they were executed.
    We are only interested in ci_checker_diagnostics executing correctly,
    not the functions that are called within.
    """

    def test_restored(
        self, mock_histogram, mock_plot_run_mean, mock_qa_image, mock_source_mask
    ):
        mock_image = MockBDSFImage()
        setattr(mock_image, "wcs_obj", Mock())

        mock_plot_run_mean.return_value = Mock()
        mock_qa_image.return_value = Mock()
        mock_histogram.return_value = Mock()
        mock_source_mask.return_value = ("source_mask", "background_mask")

        ci_checker_diagnostics(mock_image, "my_file.fits", "restored")

        assert mock_qa_image.call_count == 3
        assert mock_plot_run_mean.call_count == 3
        mock_source_mask.assert_called_once()
        mock_histogram.assert_not_called()  # only called when img is residual

    @patch(BASE_PATH + ".power_spectrum", Mock(return_value=([], [])))
    @patch(BASE_PATH + "._plot_power_spectrum", Mock())
    @patch(BASE_PATH + "._save_power_spectrum_to_csv", Mock())
    def test_residual(
        self, mock_histogram, mock_plot_run_mean, mock_qa_image, mock_source_mask
    ):
        mock_image = MockBDSFImage()
        setattr(mock_image, "wcs_obj", Mock())

        mock_plot_run_mean.return_value = Mock()
        mock_qa_image.return_value = Mock()
        mock_histogram.return_value = Mock()
        mock_source_mask.return_value = ("source_mask", "background_mask")

        ci_checker_diagnostics(mock_image, "my_file.fits", "residual")

        assert mock_qa_image.call_count == 1
        assert mock_plot_run_mean.call_count == 1
        mock_source_mask.assert_not_called()  # only called when img is restored
        mock_histogram.assert_called_once()


@patch(BASE_PATH + ".SlicedLowLevelWCS", Mock())
def test_ci_checker_diagnostics_unknown_type():
    """
    If the provided image_type is neither 'restored' nor 'residual,
    raise a ValueError.
    """
    mock_image = MockBDSFImage()
    setattr(mock_image, "wcs_obj", Mock())

    with pytest.raises(ValueError):
        ci_checker_diagnostics(mock_image, "my_file.fits", "my_weird_type")


"""
TODO:
    histogram func --> plt is mocked and testing that the call args are fine (some, not all)
    plot_with_running_mean --> 90% plotting, what isn't that's to get labels and such, not testing
    _plot_power_spectrum --> only plotting, not testing
    _save_power_spectrum_to_csv --> writing to csv, small amount of business logic, needs testing?
"""
