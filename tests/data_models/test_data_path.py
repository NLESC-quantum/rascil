"""
Data path testing
"""
import os

import pytest

from unittest.mock import patch

from rascil.data_models.parameters import rascil_path, rascil_data_path
from rascil.processing_components.util.installation_checks import check_data_directory


def test_rascil_data_path():
    result = rascil_data_path("configurations")
    assert "data/configurations" in result


@patch("rascil.data_models.parameters.rascil_path")
def test_rascil_data_path_not_exist(mock_path):
    """
    When the path we check doesn't exist, the code raises
    a FileNotFoundError (i.e. directory not found)
    """
    # rascil_data_path uses rascil_path("data")
    # here we make sure it will use a directory that does not exist.
    mock_path.return_value = rascil_path("not-exist")

    # need to remove RASCIL_DATA from the environment variables, if it exists
    # in order to produce the error correctly.
    modified_environ = {k: v for k, v in os.environ.items() if k != "RASCIL_DATA"}
    with patch.dict(os.environ, modified_environ, clear=True):
        with pytest.raises(FileNotFoundError):
            rascil_data_path("configurations")


@patch("rascil.data_models.parameters.rascil_path")
def test_check_data_directory_data_dir_not_exist_fatal_true(mock_path):
    """
    When the data directory doesn't exist, and fatal key is True,
    we raise an error with the given message.
    """
    # we have to make sure that the line
    # `canary = rascil_data_path("configurations/LOWBD2.csv")`
    # acts as if the data directory did not exist
    mock_path.return_value = rascil_path("not-exist")
    with pytest.raises(FileNotFoundError) as error:
        check_data_directory(fatal=True)

    assert str(error.value) == "The RASCIL data directory is not available - stopping"


@patch("rascil.data_models.parameters.rascil_path")
def test_check_data_directory_data_dir_not_exist_fatal_false(mock_path):
    """
    When the data directory doesn't exist, and fatal key is False,
    we log a warning with the given message.
    """
    # we have to make sure that the line
    # `canary = rascil_data_path("configurations/LOWBD2.csv")`
    # acts as if the data directory did not exist
    mock_path.return_value = rascil_path("not-exist")

    with patch("logging.Logger.warning") as mock_log:
        check_data_directory(fatal=False)
        mock_log.assert_called_with(
            "The RASCIL data directory is not available - "
            "continuing but any simulations will fail"
        )


@patch("rascil.processing_components.util.installation_checks.rascil_data_path")
def test_check_data_directory_file_not_exist_fatal_true(mock_path):
    """
    When the file we try to open doesn't exist,
    we raise a error with the given message.
    """
    mock_path.return_value = "file-not-exist"
    with pytest.raises(FileNotFoundError) as error:
        check_data_directory(fatal=True)

    assert str(error.value) == "The RASCIL data directory is not available - stopping"


@patch("rascil.processing_components.util.installation_checks.rascil_data_path")
def test_check_data_directory_file_not_exist_fatal_false(mock_path):
    """
    When the file we try to open doesn't exist,
    we raise a warning with the given message.
    """
    mock_path.return_value = "file-not-exist"
    with patch("logging.Logger.warning") as mock_log:
        check_data_directory(fatal=False)
        mock_log.assert_called_with(
            "The RASCIL data directory is not available - "
            "continuing but any simulations will fail"
        )
