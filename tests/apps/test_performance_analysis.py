""" Regression for performance analysis

"""
import logging
import pytest

from rascil.data_models.parameters import rascil_data_path

from rascil.apps.performance_analysis import cli_parser, analyser

log = logging.getLogger("rascil-logger")
log.setLevel(logging.INFO)

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
            "imaging_npixel",
            FUNCTIONS,
        ),
        (
            "line",
            "blockvis_nvis",
            FUNCTIONS,
        ),
        ("bar", "", FUNCTIONS),
        ("contour", "imaging_npixel blockvis_nvis", "invert_ng"),
    ],
)
def test_performance_analysis(mode, parameters, functions):
    """This tests the different modes of operation.

    :param mode: Mode of processing: plot or bar or contour
    :param parameters: Parameters for the test e.g. blockvis_nvis,
    :param functions: Functions for the test e.g. invert_ng
    :return:
    """

    pa_args = [
        "--mode",
        mode,
        "--performance_files",
        rascil_data_path("misc/performance_rascil_imager_360_512.json"),
        rascil_data_path("misc/performance_rascil_imager_360_1024.json"),
        rascil_data_path("misc/performance_rascil_imager_360_2048.json"),
        rascil_data_path("misc/performance_rascil_imager_360_4096.json"),
        rascil_data_path("misc/performance_rascil_imager_360_8192.json"),
    ]
    pa_args.append("--parameters")
    for parameter in parameters.split(" "):
        pa_args.append(parameter)
    pa_args.append("--functions")
    for func in functions.split(" "):
        pa_args.append(func)

    parser = cli_parser()
    args = parser.parse_args(pa_args)
    analyser(args)
