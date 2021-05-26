""" Regression for performance analysis

"""
import sys
import os
import logging
import pytest

from rascil.data_models.parameters import rascil_path, rascil_data_path

from rascil.apps.performance_analysis import cli_parser, analyser

log = logging.getLogger("rascil-logger")
log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))

FUNCTIONS = (
    "skymodel_predict_calibrate "
    "skymodel_calibrate_invert "
    "invert_ng "
    "restore_cube "
    "image_scatter_facets "
    "image_gather_facets"
)


@pytest.mark.parametrize(
    "mode, parameters, functions",
    [
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

    :param mode: Mode of processing: plot or bar or contour
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
        testfiles = [
            rascil_data_path("misc/performance_rascil_imager_360_512.json"),
            rascil_data_path("misc/performance_rascil_imager_360_1024.json"),
            rascil_data_path("misc/performance_rascil_imager_360_2048.json"),
            rascil_data_path("misc/performance_rascil_imager_360_4096.json"),
            rascil_data_path("misc/performance_rascil_imager_360_8192.json"),
        ]
    elif mode == "contour":
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
        testfiles = [
            rascil_data_path("misc/performance_rascil_imager_360_8192.json"),
        ]
    else:
        testfiles = [
            rascil_data_path("misc/performance_rascil_imager_360_8192.json"),
        ]

    parser = cli_parser()
    pa_args.append("--performance_files")
    for testfile in testfiles:
        pa_args.append(testfile)

    args = parser.parse_args(pa_args)
    filesout = analyser(args)
    # Check that the claimed output plots actually exist
    for fileout in filesout:
        f = open(fileout)
        f.close()
        if not persist:
            os.remove(fileout)
