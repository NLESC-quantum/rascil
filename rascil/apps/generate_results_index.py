"""
Generate an HTML and a Markdown file containing links
to files within a given path.

At the moment, the output index will contain links
to local files, and not online-hosted ones.
"""

import logging
import os
import datetime as dt

from rascil.data_models import rascil_path

LOGGER = logging.getLogger("rascil-logger")

# dictionary keys
LOG = "log"
FITS = "fits"
PYBDSF_SOURCE = "pybdsf_source"
RASCIL_HDF = "rascil_hdf"
STATS_PNG = "stats_png"
OTHER_FILES = "other_files"

# section titles
CATEGORY_STRINGS = {
    LOG: "Log files:",
    PYBDSF_SOURCE: "PyBDSF source files:",
    RASCIL_HDF: "RASCIL components HDF files:",
    STATS_PNG: "PNG image diagnostics files:",
    FITS: "FITS files:",
    OTHER_FILES: "Other files and sub-directories in directory:",
}

HTML_START = "<!DOCTYPE html>\n<html>\n<body>\n"
HTML_END = "</body>\n</html>\n"


def generate_html_sub_string(category_text, path, file_list):
    """
    Generate HTML string for a subsection with all files listed underneath it.

    :param category_text: text resembling the section title
    :param path: absolute path to files
    :param file_list: list of files that belong to this section/category

    :return: html-formatted string of section
    """
    html_section = f"<h2>{category_text}</h2>\n"
    for f in file_list:
        html_section = f"{html_section}\n<p><a href='file:///{path}/{f}'>{f}</a></p>\n"

    return html_section


def generate_html_file(path, sorted_file_dict):
    """
    Generate HTML file containing an index of files.

    :param path: path to directory whose files the index will contain
    :param sorted_file_dict: dictionary of files sorted into categories/sections
    """
    path = rascil_path(path)

    html_string = (
        f"{HTML_START}"
        f"<p style='font-size:22px;'> The following is an index of the list "
        f"of files generated by RASCIL continuum imaging checker and other "
        f"files found in the same directory. HTML file generated on {dt.date.today()}.\n<p>"
        f"<p style='font-size:21px;'>"
        f"<b>Contents of directory: <span style='color: red'>{path}</span></b></p>\n"
    )
    for k, v in sorted_file_dict.items():
        html_string = (
            f"{html_string}\n{generate_html_sub_string(CATEGORY_STRINGS[k], path, v)}"
        )

    html_file = open(path + "/index.html", "w")
    html_file.write(f"{html_string}" f"{HTML_END}")
    html_file.close()

    LOGGER.info("HTML file created at: %s", path + "/index.html")


def generate_md_sub_string(category_text, path, file_list):
    """
    Generate Markdown string for a subsection with all files listed underneath it.

    :param category_text: text resembling the section title
    :param path: absolute path to files
    :param file_list: list of files that belong to this section/category

    :return: markdown-formatted string of section
    """
    md_string = f"###{category_text}\n"
    for f in file_list:
        md_string = f"{md_string}\n" f"[{f}]({path}/{f})\n"

    return md_string


def generate_markdown_file(path, sorted_file_dict):
    """
    Generate Markdown file containing an index of files.

    :param path: path to directory whose files the index will contain
    :param sorted_file_dict: dictionary of files sorted into categories/sections
    """
    path = rascil_path(path)

    md_string = (
        f"<span style='font-size:18px'>The following is an index of the list "
        f"of files generated by RASCIL continuum imaging checker on and other "
        f"files found in the same directory. Markdown file generated on {dt.date.today()}. "
        f"Output directory:</span>\n"
        f"<span style='color: red; font-size:18px'>{path}</span>\n"
    )
    for k, v in sorted_file_dict.items():
        sub_string = generate_md_sub_string(CATEGORY_STRINGS[k], path, v)
        md_string = f"{md_string}\n{sub_string}"

    md_file = open(path + "/index.md", "w")
    md_file.write(md_string)
    md_file.close()

    LOGGER.info("Markdown file created at: %s", path + "/index.md")


def sort_files(path):
    """
    Sort files and sub-directories within a given path into categories.
    The categories are described in the CATEGORY_STRINGS global variable

    :param path: path to directory to check
    :return: dictionary of lists of sorted files
    """
    path = rascil_path(path)

    log_files = []
    fits_files = []
    png_files = []
    pybdsf_source_files = []
    rascil_hdf_files = []

    files = os.listdir(path)
    for f in files:
        if os.path.isdir(f"{path}/{f}"):
            continue

        if f.strip().endswith(".log"):
            log_files.append(f)

        if f.strip().endswith(".fits"):
            fits_files.append(f)

        if ".pybdsm" in f and (f.strip().endswith(".csv") or f.strip().endswith(".fits")):
            pybdsf_source_files.append(f)

        if f.strip().endswith(".hdf") or f.strip().endswith(".h5") or f.strip().endswith(".hdf5"):
            rascil_hdf_files.append(f)

        if f.strip().endswith(".png"):
            png_files.append(f)

    remaining_files = [
        x
        for x in files
        if x not in log_files
        and x not in fits_files
        and x not in pybdsf_source_files
        and x not in rascil_hdf_files
        and x not in png_files
    ]

    sorted_dict = {
        LOG: log_files,
        PYBDSF_SOURCE: pybdsf_source_files,
        RASCIL_HDF: rascil_hdf_files,
        STATS_PNG: png_files,
        FITS: fits_files,
        OTHER_FILES: remaining_files,
    }

    return sorted_dict


def create_index(path):
    LOGGER.info(
        "Generating index HTML and Markdown files for contents of directory: %s",
        rascil_path(path),
    )

    sorted_files = sort_files(path)
    generate_markdown_file(path, sorted_files)
    generate_html_file(path, sorted_files)
