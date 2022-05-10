""" Test for rascil_sensitivity

"""
import os
import logging
import pandas as pd

import pytest

from rascil.data_models import rascil_path
from rascil.apps.rascil_sensitivity import cli_parser, calculate_sensitivity

log = logging.getLogger("rascil-logger")
log.setLevel(logging.WARNING)

default_run = True


@pytest.mark.parametrize(
    "enabled, tag, use_dask, frequency, npixel, cellsize, weighting, rmax, robustnesses, tapers",
    [
        (
            default_run,
            "B1LOW",
            True,
            0.350e9,
            512,
            6e-7,
            "robust",
            1e4,
            "-2 -1 0 1 2",
            "0.0 5e-6 1e-5",
        ),
        (
            default_run,
            "B1LOW_NATURAL",
            True,
            0.350e9,
            512,
            6e-7,
            "natural",
            1e4,
            "0",
            "0.0 5e-6 1e-5",
        ),
        (
            default_run,
            "B1LOW_UNIFORM",
            True,
            0.350e9,
            512,
            6e-7,
            "uniform",
            1e4,
            "0",
            "0.0 5e-6 1e-5",
        ),
        (
            default_run,
            "B2",
            True,
            1.36e9,
            1024,
            2e-7,
            "robust",
            1e4,
            "-2 -1 0 1 2",
            "0.0 1.2e-6",
        ),
        (
            default_run,
            "B2_NATURAL",
            True,
            1.36e9,
            1024,
            2e-7,
            "natural",
            1e4,
            "0",
            "0.0 1.2e-6",
        ),
        (
            default_run,
            "B2_UNIFORM",
            True,
            1.36e9,
            1024,
            2e-7,
            "uniform",
            1e4,
            "0",
            "0.0 1.2e-6",
        ),
    ],
)
def test_rascil_sensitivity(
    enabled,
    tag,
    use_dask,
    frequency,
    npixel,
    cellsize,
    weighting,
    rmax,
    robustnesses,
    tapers,
):
    """

    :param enabled: Turn this test on?
    :param tag: Tag for files generated
    :param use_dask: Use dask for processing. Set to False for debugging
    :param frequency: Frequency of test images (Hz)
    :param npixel: Number of pixels in test images
    :param cellsize: Cellsize of test images (rad)
    :param weighting: type of weighting
    :param rmax: Maximum distance of dish from array centre (m)
    :param robustnesses: Set of robustness values to test
    :param tapers: Set of tapers to try (scale size in image in rad)
    """
    robustnesses = robustnesses.split(" ")
    tapers = tapers.split(" ")

    persist = os.getenv("RASCIL_PERSIST", False)

    if not enabled:
        return True

    results = rascil_path(f"test_results/{tag}")
    sensitivity_args = [
        "--frequency",
        f"{frequency}",
        "--imaging_cellsize",
        f"{cellsize}",
        "--imaging_npixel",
        f"{npixel}",
        "--imaging_weighting",
        f"{weighting}",
        "--rmax",
        f"{rmax}",
        "--results",
        results,
        "--verbose",
        f"{persist}",
    ]
    sensitivity_args.append("--imaging_taper")
    for taper in tapers:
        sensitivity_args.append(taper)

    if weighting not in ["uniform", "natural"]:
        sensitivity_args.append("--imaging_robustness")
        for robust in robustnesses:
            sensitivity_args.append(robust)

    parser = cli_parser()
    args = parser.parse_args(sensitivity_args)

    results_file = calculate_sensitivity(args)

    if os.path.exists(results_file) is False:
        log.error(f"Error: No results file {results_file} found.")

    df = pd.read_csv(results_file)

    # Check the shape of the DataFrame and the column names
    if weighting in ["uniform", "natural"]:
        nrows = len(tapers)
    else:
        nrows = len(tapers * (len(robustnesses)))

    assert len(df) == nrows
    assert len(df.columns) == 21

    columns = [
        "weighting",
        "robustness",
        "taper",
        "cleanbeam_bmaj",
        "cleanbeam_bmin",
        "cleanbeam_bpa",
        "sum_weights",
        "psf_shape",
        "psf_size",
        "psf_max",
        "psf_min",
        "psf_maxabs",
        "psf_rms",
        "psf_sum",
        "psf_medianabs",
        "psf_medianabsdevmedian",
        "psf_median",
        "pss",
        "sa",
        "sbs",
        "tb",
    ]
    for col in df.columns:
        assert col in columns
    for col in columns:
        assert col in df.columns
