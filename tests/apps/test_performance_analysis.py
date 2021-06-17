""" Regression for performance analysis

"""
import sys
import os
import logging

import numpy.testing
import pytest

from rascil.data_models.parameters import rascil_path, rascil_data_path

from rascil.apps.performance_analysis import cli_parser, analyser, fit_2d_plane

log = logging.getLogger("rascil-logger")
log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))

FUNCTIONS = (
    "skymodel_predict_calibrate "
    "skymodel_calibrate_invert "
    "create_blockvisibility_from_ms "
    "imaging_deconvolve "
    "invert_ng "
    "restore_cube "
    "image_scatter_facets "
    "image_gather_facets"
)


@pytest.mark.parametrize(
    "mode, parameters, functions",
    [
        ("fit", "imaging_npixel_sq blockvis_nvis", "summary"),
        (
            "line",
            "imaging_npixel_sq",
            FUNCTIONS,
        ),
        (
            "memory_histogram",
            "",
            FUNCTIONS,
        ),
        ("summary", "imaging_npixel_sq blockvis_nvis", "summary"),
        (
            "line",
            "imaging_npixel_sq",
            FUNCTIONS,
        ),
        ("bar", "", FUNCTIONS),
        ("contour", "imaging_npixel_sq blockvis_nvis", "invert_ng"),
    ],
)
def test_performance_analysis(mode, parameters, functions):
    """This tests the different modes of operation.

    :param mode: Mode of processing: plot or bar or contour or memory_histogram or fit
    :param parameters: Parameters for the test e.g. blockvis_nvis,
    :param functions: Functions for the test e.g. invert_ng
    :return:
    """

    persist = os.getenv("RASCIL_PERSIST", False)

    pa_args = [
        "--mode",
        mode,
    ]
    pa_args.append("--parameters")
    for parameter in parameters.split(" "):
        pa_args.append(parameter)
    pa_args.append("--functions")
    for func in functions.split(" "):
        pa_args.append(func)
    pa_args.append("--results")
    pa_args.append(rascil_path("test_results"))

    if mode == "line":
        cli_arg = "--performance_files"
        testfiles = [
            rascil_data_path("misc/performance_rascil_imager_360_512.json"),
            rascil_data_path("misc/performance_rascil_imager_360_1024.json"),
            rascil_data_path("misc/performance_rascil_imager_360_2048.json"),
            rascil_data_path("misc/performance_rascil_imager_360_4096.json"),
            rascil_data_path("misc/performance_rascil_imager_360_8192.json"),
        ]
    elif mode == "contour" or mode == "summary" or mode == "fit":
        cli_arg = "--performance_files"
        testfiles = [
            rascil_data_path("misc/performance_rascil_imager_360_512.json"),
            rascil_data_path("misc/performance_rascil_imager_360_1024.json"),
            rascil_data_path("misc/performance_rascil_imager_360_2048.json"),
            rascil_data_path("misc/performance_rascil_imager_360_4096.json"),
            rascil_data_path("misc/performance_rascil_imager_360_8192.json"),
            rascil_data_path("misc/performance_rascil_imager_720_512.json"),
            rascil_data_path("misc/performance_rascil_imager_720_1024.json"),
            rascil_data_path("misc/performance_rascil_imager_720_2048.json"),
            rascil_data_path("misc/performance_rascil_imager_720_4096.json"),
            rascil_data_path("misc/performance_rascil_imager_720_8192.json"),
            rascil_data_path("misc/performance_rascil_imager_1440_512.json"),
            rascil_data_path("misc/performance_rascil_imager_1440_1024.json"),
            rascil_data_path("misc/performance_rascil_imager_1440_2048.json"),
            rascil_data_path("misc/performance_rascil_imager_1440_4096.json"),
            rascil_data_path("misc/performance_rascil_imager_1440_8192.json"),
            rascil_data_path("misc/performance_rascil_imager_2880_512.json"),
            rascil_data_path("misc/performance_rascil_imager_2880_1024.json"),
            rascil_data_path("misc/performance_rascil_imager_2880_2048.json"),
            rascil_data_path("misc/performance_rascil_imager_2880_4096.json"),
            rascil_data_path("misc/performance_rascil_imager_2880_8192.json"),
        ]
    elif mode == "bar":
        cli_arg = "--performance_files"
        testfiles = [
            rascil_data_path("misc/performance_rascil_imager_360_8192.json"),
        ]
    elif mode == "memory_histogram":
        cli_arg = "--memory_file"
        testfiles = [
            rascil_data_path("misc/performance_rascil_imager_360_8192.csv"),
        ]
    else:
        cli_arg = "--performance_files"
        testfiles = [
            rascil_data_path("misc/performance_rascil_imager_360_8192.json"),
        ]

    parser = cli_parser()
    pa_args.append(cli_arg)
    for testfile in testfiles:
        pa_args.append(testfile)

    args = parser.parse_args(pa_args)
    results = analyser(args)

    if mode != "fit":
        # Check that the claimed output plots actually exist
        for fileout in results:
            f = open(fileout)
            f.close()
            if not persist:
                os.remove(fileout)
    else:
        numpy.testing.assert_almost_equal(
            results["duration"]["p"], 70.6668772599627e-6, err_msg=str(results)
        )
        numpy.testing.assert_almost_equal(
            results["duration"]["q"], 8.698426561359968e-6, err_msg=str(results)
        )
        numpy.testing.assert_almost_equal(
            results["processor_time"]["p"], 216.28881124458044e-6, err_msg=str(results)
        )
        numpy.testing.assert_almost_equal(
            results["processor_time"]["q"], 858.9358920277938e-6, err_msg=str(results)
        )
        numpy.testing.assert_almost_equal(
            results["speedup"]["p"], -0.47391335429625137e-6, err_msg=str(results)
        )
        numpy.testing.assert_almost_equal(
            results["speedup"]["q"], 1.3078320080729133e-6, err_msg=str(results)
        )


@pytest.mark.parametrize(
    "p, q",
    [(1.0, 2.0), (0.0, 2.0), (-1001.0, 2.0), (-1e7, 1.0), (-1e7, 2e7)],
)
def test_fit_2d_plane(p, q):
    """Test the fit of a 2D plane z = p * x + q + y for a range of inputs"""

    def sim(p_actual, q_actual):
        x = numpy.linspace(1e5, 1e6, 10)
        y = numpy.linspace(1e5, 1e6, 10)
        xx, yy = numpy.meshgrid(x, y)
        return xx, yy, p_actual * xx + q_actual * yy

    x, y, z = sim(p, q)

    p_estimate, q_estimate = fit_2d_plane(x, y, z)

    numpy.testing.assert_almost_equal(p, p_estimate)
    numpy.testing.assert_almost_equal(q, q_estimate)
