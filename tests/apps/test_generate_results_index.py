import os
import shutil
from unittest.mock import patch, Mock

from rascil.apps.generate_results_index import (
    generate_html_sub_string,
    generate_md_sub_string,
    generate_html_file,
    LOG,
    FITS,
    STATS_PNG,
    generate_markdown_file,
    sort_files,
    OTHER_FILES,
)
from rascil.data_models import rascil_path


def test_generate_html_sub_string_one_file():
    """
    HTML-formatted string correctly contains the one file
    in the input list with the correct path give in "path"
    """
    path = "myHome/test_results"
    file_list = ["myFig.png"]
    section_title = "These are my figures"

    expected_string = (
        "<h2>These are my figures</h2>\n\n"
        "<p><a href='file:///myHome/test_results/myFig.png'>myFig.png</a></p>\n"
    )

    result = generate_html_sub_string(section_title, path, file_list)
    assert result == expected_string


def test_generate_html_sub_string_multiple_files():
    """
    HTML-formatted string correctly contains all of the files
    in the input list with the correct path give in "path"
    """
    path = "myHome/test_results"
    file_list = ["myFig.png", "mySecondFig.png"]
    section_title = "These are my figures"

    expected_string = (
        "<h2>These are my figures</h2>\n\n"
        "<p><a href='file:///myHome/test_results/myFig.png'>myFig.png</a></p>\n\n"
        "<p><a href='file:///myHome/test_results/mySecondFig.png'>mySecondFig.png</a></p>\n"
    )

    result = generate_html_sub_string(section_title, path, file_list)
    assert result == expected_string


def test_generate_html_sub_string_no_files():
    """
    HTML-formatted string only contains the section title,
    when input file list is empty."
    """
    path = "myHome/test_results"
    file_list = []
    section_title = "These are my figures"

    expected_string = "<h2>These are my figures</h2>\n"

    result = generate_html_sub_string(section_title, path, file_list)
    assert result == expected_string


def test_generate_md_sub_string_one_file():
    """
    Markdown-formatted string correctly contains the one file
    in the input list with the correct path give in "path"
    """
    path = "myHome/test_results"
    file_list = ["myFig.png"]
    section_title = "These are my figures"

    expected_string = (
        "###These are my figures\n\n" "[myFig.png](myHome/test_results/myFig.png)\n"
    )

    result = generate_md_sub_string(section_title, path, file_list)
    assert result == expected_string


def test_generate_md_sub_string_multiple_files():
    """
    Markdown-formatted string correctly contains all of the files
    in the input list with the correct path give in "path"
    """
    path = "myHome/test_results"
    file_list = ["myFig.png", "mySecondFig.png"]
    section_title = "These are my figures"

    expected_string = (
        "###These are my figures\n\n"
        "[myFig.png](myHome/test_results/myFig.png)\n\n"
        "[mySecondFig.png](myHome/test_results/mySecondFig.png)\n"
    )

    result = generate_md_sub_string(section_title, path, file_list)
    assert result == expected_string


def test_generate_md_sub_string_no_files():
    """
    Markdown-formatted string only contains the section title,
    when input file list is empty."
    """
    path = "myHome/test_results"
    file_list = []
    section_title = "These are my figures"

    expected_string = "###These are my figures\n"

    result = generate_md_sub_string(section_title, path, file_list)
    assert result == expected_string


@patch("rascil.apps.generate_results_index.open")
def test_generate_html_file_headers_only(mock_open):
    """
    HTML file correctly generated, with section titles only,
    when input file lists are empty.
    """
    path = "test_results"
    files_dict = {LOG: [], FITS: [], STATS_PNG: []}
    mock_open.return_value = Mock()

    # we expect the following to be part of the full string with which .write was called
    # the lists are empty so we only expect the headers to be present
    expected_sub_string = "<h2>Log files:</h2>\n\n<h2>FITS files:</h2>\n\n<h2>PNG image diagnostics files:</h2>"

    generate_html_file(path, files_dict)

    # the string that .write was called with
    call_arg_string = mock_open.return_value.write.call_args_list[0][0][0]

    mock_open.return_value.write.assert_called_once()
    assert expected_sub_string in call_arg_string


@patch("rascil.apps.generate_results_index.open")
def test_generate_html_file_headers_and_content(mock_open):
    """
    HTML file correctly generated, including all of the section titles
    present in the input dictionary, and all the files that are
    in the file lists (values of the dict)
    """
    path = "test_results"
    files_dict = {LOG: ["myLog.log"], FITS: [], STATS_PNG: ["myFig.png"]}
    mock_open.return_value = Mock()

    # we expect the following to be part of the full string with which .write was called
    # the three headers, and the contents of the two lists are expected
    expected_sub_string = (
        f"<h2>Log files:</h2>\n\n<p><a href='file:///{rascil_path(path)}/myLog.log'>"
        f"myLog.log</a></p>\n\n<h2>FITS files:</h2>\n\n"
        f"<h2>PNG image diagnostics files:</h2>\n\n"
        f"<p><a href='file:///{rascil_path(path)}/myFig.png'>myFig.png</a></p>\n"
    )

    generate_html_file(path, files_dict)

    # the string that .write was called with
    call_arg_string = mock_open.return_value.write.call_args_list[0][0][0]

    mock_open.return_value.write.assert_called_once()
    assert expected_sub_string in call_arg_string


@patch("rascil.apps.generate_results_index.open")
def test_generate_markdown_file_headers_only(mock_open):
    """
    Markdown file correctly generated, with section titles only,
    when input file lists are empty.
    """
    path = "test_results"
    files_dict = {LOG: [], FITS: [], STATS_PNG: []}
    mock_open.return_value = Mock()

    # we expect the following to be part of the full string with which .write was called
    # the lists are empty so we only expect the headers to be present
    expected_sub_string = (
        "###Log files:\n\n###FITS files:\n\n###PNG image diagnostics files:"
    )

    generate_markdown_file(path, files_dict)

    # the string that .write was called with
    call_arg_string = mock_open.return_value.write.call_args_list[0][0][0]

    mock_open.return_value.write.assert_called_once()
    assert expected_sub_string in call_arg_string


@patch("rascil.apps.generate_results_index.open")
def test_generate_markdown_file_headers_and_content(mock_open):
    """
    Markdown file correctly generated, including all of the section titles
    present in the input dictionary, and all the files that are
    in the file lists (values of the dict)
    """
    path = "test_results"
    files_dict = {LOG: ["myLogs.log"], FITS: [], STATS_PNG: ["myFig.png"]}
    mock_open.return_value = Mock()

    # we expect the following to be part of the full string with which .write was called
    # the three headers, and the contents of the two lists are expected
    expected_sub_string = (
        f"###Log files:\n\n[myLogs.log]({rascil_path(path)}/myLogs.log)\n\n"
        "###FITS files:\n\n###PNG image diagnostics files:\n\n"
        f"[myFig.png]({rascil_path(path)}/myFig.png)\n"
    )

    generate_markdown_file(path, files_dict)

    # the string that .write was called with
    call_arg_string = mock_open.return_value.write.call_args_list[0][0][0]

    mock_open.return_value.write.assert_called_once()
    assert expected_sub_string in call_arg_string


def test_sort_files_empty_dir():
    """
    If directory is empty, the values of the returned dictionary will be empty lists.
    """
    path = rascil_path("tests/apps/myFakeDirectoryForTests")
    os.mkdir(path)

    result = sort_files(path)
    os.rmdir(path)

    assert list(result.values()) == [[], [], [], [], [], []]


def test_sort_files_some_files():
    """
    If directory contains files, all of these files will appear under
    at least one dictionary key.
    """
    path = rascil_path("tests/apps/myFakeDirectoryForTests")
    os.mkdir(path)  # create temporary new directory
    # add files to new directory
    open(f"{path}/myFig.png", "w")
    open(f"{path}/some_file.txt", "w")
    open(f"{path}/firstLog.log", "w")
    open(f"{path}/secondLog.log", "w")

    result = sort_files(path)
    shutil.rmtree(path)  # delete directory with files in it

    assert result[STATS_PNG] == ["myFig.png"]
    assert result[OTHER_FILES] == ["some_file.txt"]
    assert sorted(result[LOG]) == sorted(["firstLog.log", "secondLog.log"])
