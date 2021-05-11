""" Unit processing_components for rascil-image_check

"""
import logging

import pytest

from rascil.apps.rascil_image_check import cli_parser, image_check
from rascil.data_models import rascil_data_path

log = logging.getLogger("rascil-logger")
log.setLevel(logging.INFO)
default_run = True


@pytest.mark.parametrize(
    "enabled, stat, stat_min, stat_max, result",
    [
        (default_run, "max", 0.0, 1.2, 0),
        (default_run, "min", 0.0, 0.0, 0),
        (default_run, "max", 0.0, 0.2, 1),
        (default_run, "rms", 0.0, 0.2, 0),
    ],
)
def test_image_check(enabled, stat, stat_min, stat_max, result):
    """Check that the results of the check are as expected

    :param enabled: Turn this test on?
    :param stat: Statistic to be tested
    :param stat_min: Minimum value
    :param stat_max: Maximum value
    :param result: Expected result 0 for success, 1 for failure
    :return:
    """
    if not enabled:
        return True

    fits_image = rascil_data_path("models/M31_canonical.model.fits")

    check_args = [
        "--stat",
        f"{stat}",
        "--image",
        f"{fits_image}",
        "--stat",
        f"{stat}",
        "--max",
        f"{stat_max}",
        "--min",
        f"{stat_min}",
    ]

    parser = cli_parser()
    args = parser.parse_args(check_args)
    r = image_check(args)
    if r != result:
        raise ValueError(f"Result of check {r} not as expected: {result}")
