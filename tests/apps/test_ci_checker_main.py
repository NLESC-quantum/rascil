""" Regression for continuum imaging checker

    The script mainly tests two things:
    1) the BDSF source finder
    2) the matching algorithm

    Input parameters:
    param cellsize: Cell size of each pixel in the image
    param npixel: Number of pixels for the generated image
    param flux_limit: Threshold for source selection
    param insert_method: Method of interpolation for inserting sources into image
    param noise: amount of noise added onto the image
    param tag: Tag to keep track of the relavant files and images.

"""
import logging
import sys
import os
import glob

import astropy.units as u
import numpy
import pytest
from astropy.coordinates import SkyCoord
from numpy.random import default_rng

from rascil.apps.ci_checker_main import (
    cli_parser,
    analyze_image,
)
from rascil.data_models.parameters import rascil_path
from rascil.data_models.polarisation import PolarisationFrame
from rascil.data_models.data_model_helpers import export_skycomponent_to_hdf5
from rascil.processing_components.image import (
    create_image,
    export_image_to_fits,
    smooth_image,
)
from rascil.processing_components.simulation import (
    create_mid_simulation_components,
    find_pb_width_null,
)
from rascil.processing_components.skycomponent import (
    insert_skycomponent,
    find_skycomponent_matches,
)

log = logging.getLogger("rascil-logger")


@pytest.mark.parametrize(
    "cellsize, npixel, flux_limit, insert_method, noise, tag",
    [
        (
            0.0001,
            512,
            0.001,
            "Nearest",
            0.00003,
            "nearest_npixel512_noise0.00003_flux0.001",
        ),
        (
            0.0001,
            512,
            0.0001,
            "Nearest",
            0.00003,
            "nearest_npixel512_noise0.00003_flux0.0001",
        ),
        (
            0.0001,
            512,
            0.001,
            "Nearest",
            0.0003,
            "nearest_npixel512_noise0.00003_flux0.001",
        ),
        (
            0.0001,
            512,
            0.001,
            "Lanczos",
            0.0003,
            "lanczos_npixel512_noise0.00003_flux0.001",
        ),
        (
            0.0001,
            1024,
            0.001,
            "Nearest",
            0.00003,
            "nearest_npixel1024_noise0.00003_flux0.001",
        ),
    ],
)
def test_continuum_imaging_checker(
    cellsize, npixel, flux_limit, insert_method, noise, tag
):

    # Set true if we want to save the outputs
    persist = os.getenv("RASCIL_PERSIST", False)

    # set up
    image_frequency = numpy.array([1.0e9])
    phasecentre = SkyCoord(
        ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
    )
    hwhm_deg, null_az_deg, null_el_deg = find_pb_width_null(
        pbtype="MID", frequency=image_frequency
    )

    nchan = image_frequency.shape[0]
    central_freq = image_frequency[int(nchan // 2)]

    hwhm = hwhm_deg * numpy.pi / 180.0
    fov_deg = 8.0 * 1.36e9 / central_freq
    pb_npixel = 256
    d2r = numpy.pi / 180.0
    pb_cellsize = d2r * fov_deg / pb_npixel
    pbradius = 1.5
    pbradius = pbradius * hwhm

    original_components = create_mid_simulation_components(
        phasecentre,
        image_frequency,
        flux_limit,
        pbradius,
        pb_npixel,
        pb_cellsize,
        show=False,
        fov=10,
        apply_pb=True,
    )

    components = original_components[0]
    components = sorted(components, key=lambda cmp: numpy.max(cmp.direction.ra))

    comp_file = rascil_path(f"test_results/test_ci_checker_{tag}.hdf")
    export_skycomponent_to_hdf5(components, comp_file)

    txtfile = rascil_path(f"test_results/test_ci_checker_{tag}.txt")
    f = open(txtfile, "w")
    f.write(
        "# RA(deg), Dec(deg), I (Jy), Q (Jy), U (Jy), V (Jy), Ref. freq. (Hz), Spectral index\n"
    )
    for cmp in components:
        coord_ra = cmp.direction.ra.degree
        coord_dec = cmp.direction.dec.degree
        f.write(
            "%.6f, %.6f, %10.6e, %10.6e, %10.6e, %10.6e, %10.6e, %10.6e \n"
            % (coord_ra, coord_dec, cmp.flux[0], 0.0, 0.0, 0.0, central_freq, 0.0)
        )
    f.close()

    model = create_image(
        npixel=npixel,
        cellsize=cellsize,
        phasecentre=phasecentre,
        frequency=image_frequency,
        polarisation_frame=PolarisationFrame("stokesI"),
    )

    model = insert_skycomponent(model, components, insert_method=insert_method)

    if noise > 0.0:
        rng = default_rng(1805550721)
        model["pixels"].data += rng.normal(0.0, noise, model["pixels"].data.shape)

    model = smooth_image(model, width=1.0, normalise=False)

    tagged_file = rascil_path(f"test_results/test_ci_checker_{tag}.fits")
    export_image_to_fits(model, tagged_file)
    tagged_file_residual = None

    parser = cli_parser()
    args = parser.parse_args(
        [
            "--ingest_fitsname_restored",
            tagged_file,
            "--finder_beam_maj",
            f"{numpy.rad2deg(cellsize)}",
            "--finder_beam_min",
            f"{numpy.rad2deg(cellsize)}",
            "--check_source",
            "True",
            "--plot_source",
            "True",
            "--input_source_format",
            "external",
            "--input_source_filename",
            comp_file,  # txtfile
            "--match_sep",
            "1.0e-3",
        ]
    )

    out, matches_found = analyze_image(args)

    # check results directly
    sorted_comp = sorted(out, key=lambda cmp: numpy.max(cmp.direction.ra))
    log.info("Identified components:")
    for cmp in sorted_comp:
        coord_ra = cmp.direction.ra.degree
        coord_dec = cmp.direction.dec.degree
        log.info("%.6f, %.6f, %10.6e \n" % (coord_ra, coord_dec, cmp.flux[0]))

    assert len(out) <= len(components)
    log.info(
        "BDSF expected to find %d sources, but found %d sources"
        % (len(components), len(out))
    )
    matches_expected = find_skycomponent_matches(out, components, tol=1e-3)
    log.info("Found matches as follows.")
    log.info("BDSF Original Separation")
    for match in matches_expected:
        log.info("%d %d %10.6e" % (match[0], match[1], match[2]))

    numpy.testing.assert_array_almost_equal(matches_found, matches_expected)

    assert os.path.exists(
        rascil_path(f"test_results/test_ci_checker_{tag}_restored_plot.png")
    )
    assert os.path.exists(
        rascil_path(f"test_results/test_ci_checker_{tag}_sources_plot.png")
    )
    assert os.path.exists(
        rascil_path(f"test_results/test_ci_checker_{tag}_background_plot.png")
    )
    if tagged_file_residual is not None:
        assert os.path.exists(
            rascil_path(f"test_results/test_ci_checker_{tag}_residual_hist.png")
        )
        assert os.path.exists(
            rascil_path(
                f"test_results/test_ci_checker_{tag}_residual_power_spectrum.png"
            )
        )

    assert os.path.exists(
        rascil_path(f"test_results/test_ci_checker_{tag}_position_value.png")
    )
    assert os.path.exists(
        rascil_path(f"test_results/test_ci_checker_{tag}_position_error.png")
    )
    assert os.path.exists(
        rascil_path(f"test_results/test_ci_checker_{tag}_position_distance.png")
    )
    assert os.path.exists(
        rascil_path(f"test_results/test_ci_checker_{tag}_flux_value.png")
    )
    assert os.path.exists(
        rascil_path(f"test_results/test_ci_checker_{tag}_flux_ratio.png")
    )
    assert os.path.exists(
        rascil_path(f"test_results/test_ci_checker_{tag}_flux_histogram.png")
    )

    assert os.path.exists(
        rascil_path(f"test_results/test_ci_checker_{tag}_position_quiver.png")
    )

    assert os.path.exists(
        rascil_path(f"test_results/test_ci_checker_{tag}_gaussian_beam_position.png")
    )
    # test that create_index() generates the html and md files,
    # at the end of analyze_image()
    assert os.path.exists(rascil_path("test_results/index.html"))
    assert os.path.exists(rascil_path("test_results/index.md"))

    # clean up directory
    if persist is False:
        imglist = glob.glob(rascil_path(f"test_results/test_ci_checker_{tag}*"))
        for f in imglist:
            os.remove(f)
        try:
            os.remove(rascil_path("test_results/index.html"))
            os.remove(rascil_path("test_results/index.md"))
        except OSError:
            pass
