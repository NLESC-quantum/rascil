""" Regression/integration test for continuum imaging checker

    The script mainly tests two things:
    1) the BDSF source finder
    2) the matching algorithm

    Input parameters:
    param cellsize: Cell size of each pixel in the image
    param npixel: Number of pixels for the generated image
    param nchan: Number of frequency channels in the image
    param flux_limit: Threshold for source selection
    param insert_method: Method of interpolation for inserting sources into image
    param noise: amount of noise added onto the image
    param tag: Tag to keep track of the relavant files and images.

"""
import logging
import os
import glob

import astropy.units as u
import numpy
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from numpy.random import default_rng

from rascil.apps.imaging_qa_main import (
    cli_parser,
    analyze_image,
)
from rascil.data_models.parameters import rascil_path
from rascil.data_models.polarisation import PolarisationFrame
from rascil.data_models.data_model_helpers import export_skycomponent_to_hdf5
from rascil.processing_components.imaging.primary_beams import create_pb
from rascil.processing_components.image import (
    create_image,
    export_image_to_fits,
    restore_cube,
)
from rascil.processing_components.simulation import (
    create_mid_simulation_components,
    find_pb_width_null,
)
from rascil.processing_components.skycomponent import (
    insert_skycomponent,
    find_skycomponent_matches,
    fit_skycomponent_spectral_index,
    apply_beam_to_skycomponent,
    copy_skycomponent,
)

log = logging.getLogger("rascil-logger")
log.setLevel(logging.INFO)
logging.getLogger("fit_skycomponent_spectral_index").setLevel(logging.INFO)


@pytest.mark.parametrize(
    "use_dask, cellsize, npixel, nchan, flux_limit, insert_method, noise, tag",
    [
        (
            "True",
            0.0001,
            512,
            1,
            0.001,
            "Nearest",
            0.00003,
            "nearest_npixel512_nchan1_noise0.00003_flux0.001",
        ),
        (
            "True",
            0.0001,
            1024,
            1,
            0.001,
            "Nearest",
            0.00003,
            "nearest_npixel1024_nchan1_noise0.00003_flux0.001",
        ),
        (
            "True",
            0.0001,
            512,
            8,
            0.001,
            "Nearest",
            0.00003,
            "nearest_npixel512_nchan8_noise0.00003_flux0.001",
        ),
        (
            "True",
            0.0001,
            1024,
            8,
            0.001,
            "Nearest",
            0.000001,
            "nearest_npixel1024_nchan8_noise0.000001_flux0.001",
        ),
        (
            "True",
            0.0001,
            512,
            1,
            0.0001,
            "Nearest",
            0.00003,
            "nearest_npixel512_nchan1_noise0.00003_flux0.0001",
        ),
        (
            "True",
            0.0001,
            512,
            1,
            0.001,
            "Lanczos",
            0.00003,
            "lanczos_npixel512_nchan1_noise0.00003_flux0.001",
        ),
        (
            "True",
            0.0001,
            512,
            1,
            0.001,
            "Nearest",
            0.0003,
            "nearest_npixel512_nchan1_noise0.0003_flux0.001",
        ),
    ],
)
def test_continuum_imaging_checker(
    use_dask, cellsize, npixel, nchan, flux_limit, insert_method, noise, tag
):

    # Set true if we want to save the outputs
    persist = os.getenv("RASCIL_PERSIST", True)

    # set up
    phasecentre = SkyCoord(
        ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
    )
    if nchan == 1:
        image_frequency = numpy.array([1.0e9])
    else:
        image_frequency = numpy.linspace(0.8e9, 1.2e9, nchan)

    central_freq = image_frequency[int(nchan // 2)]

    clean_beam = {
        "bmaj": numpy.rad2deg(5.0 * cellsize),
        "bmin": numpy.rad2deg(5.0 * cellsize) / 2.0,
        "bpa": 0.0,
    }

    # Add primary beam
    hwhm_deg, null_az_deg, null_el_deg = find_pb_width_null(
        pbtype="MID",
        frequency=numpy.array([central_freq]),
    )
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
        apply_pb=False,
    )

    # Apply primary beam and export to sensitivity image
    pbmodel = create_image(
        npixel=pb_npixel,
        cellsize=pb_cellsize,
        phasecentre=phasecentre,
        frequency=image_frequency,
        polarisation_frame=PolarisationFrame("stokesI"),
    )

    pb = create_pb(pbmodel, "MID", pointingcentre=phasecentre, use_local=False)
    components_with_pb = apply_beam_to_skycomponent(original_components[0], pb)

    sensitivity_file = rascil_path(
        f"test_results/test_imaging_qa_{tag}_sensitivity.fits"
    )
    export_image_to_fits(pb, sensitivity_file)

    # Write out the original components
    components = original_components[0]
    components = sorted(components, key=lambda cmp: numpy.max(cmp.direction.ra))

    comp_file = rascil_path(f"test_results/test_imaging_qa_{tag}.hdf")
    export_skycomponent_to_hdf5(components, comp_file)

    # Create restored image
    model = create_image(
        npixel=npixel,
        cellsize=cellsize,
        phasecentre=phasecentre,
        frequency=image_frequency,
        polarisation_frame=PolarisationFrame("stokesI"),
    )

    model = insert_skycomponent(model, components_with_pb, insert_method=insert_method)

    if noise > 0.0:
        rng = default_rng(1805550721)
        model["pixels"].data += rng.normal(0.0, noise, model["pixels"].data.shape)

    model = restore_cube(model, clean_beam=clean_beam)
    model.attrs["clean_beam"] = clean_beam

    restored_file = rascil_path(f"test_results/test_imaging_qa_{tag}.fits")
    export_image_to_fits(model, restored_file)

    # Generate residual file: No skycomponents, just noise
    residual_model = create_image(
        npixel=npixel,
        cellsize=cellsize,
        phasecentre=phasecentre,
        frequency=image_frequency,
        polarisation_frame=PolarisationFrame("stokesI"),
    )

    if noise > 0.0:
        residual_model["pixels"].data += rng.normal(
            0.0, noise, residual_model["pixels"].data.shape
        )

    residual_model = restore_cube(residual_model, clean_beam=clean_beam)
    residual_model.attrs["clean_beam"] = clean_beam

    residual_file = rascil_path(f"test_results/test_imaging_qa_{tag}_residual.fits")
    export_image_to_fits(residual_model, residual_file)

    # Create frequency moment image
    taylor_model = create_image(
        npixel=npixel,
        cellsize=cellsize,
        phasecentre=phasecentre,
        frequency=image_frequency,
        polarisation_frame=PolarisationFrame("stokesI"),
    )
    taylor_components = copy_skycomponent(components_with_pb)
    spec_indx = -0.7
    for comp in taylor_components:
        comp.flux = comp.flux * spec_indx

    taylor_model = insert_skycomponent(
        taylor_model, taylor_components, insert_method=insert_method
    )

    if noise > 0.0:
        taylor_model["pixels"].data += rng.normal(
            0.0, noise, taylor_model["pixels"].data.shape
        )

    taylor_file = rascil_path(f"test_results/test_imaging_qa_{tag}_taylor1.fits")
    export_image_to_fits(taylor_model, taylor_file)

    parser = cli_parser()
    args = parser.parse_args(
        [
            "--ingest_fitsname_restored",
            restored_file,
            "--ingest_fitsname_residual",
            residual_file,
            "--ingest_fitsname_sensitivity",
            sensitivity_file,
            "--ingest_fitsname_moment",
            rascil_path(f"test_results/test_imaging_qa_{tag}"),
            "--check_source",
            "True",
            "--plot_source",
            "True",
            "--input_source_filename",
            comp_file,  # hdffile
            "--match_sep",
            "1.0e-4",
            "--use_frequency_moment",
            "True",
            "--perform_diagnostics",
            "True",
            "--apply_primary",
            "True",
            "--savefits_rmsim",
            "True",
            "--use_dask",
            use_dask,
        ]
    )

    out, matches_found = analyze_image(args)

    # check results directly
    sorted_comp = sorted(out, key=lambda cmp: numpy.max(cmp.direction.ra))
    log.debug("Identified components:")
    for cmp in sorted_comp:
        coord_ra = cmp.direction.ra.degree
        coord_dec = cmp.direction.dec.degree
        log.debug("%.6f, %.6f, %10.6e \n" % (coord_ra, coord_dec, cmp.flux[0]))

    assert len(out) <= len(components)
    log.info(
        "BDSF expected to find %d sources, but found %d sources"
        % (len(components), len(out))
    )
    matches_expected = find_skycomponent_matches(out, components, tol=1e-4)
    log.debug("Found matches as follows.")
    log.debug("BDSF Original Separation")
    for match in matches_expected:
        log.debug("%d %d %10.6e" % (match[0], match[1], match[2]))

    numpy.testing.assert_array_almost_equal(matches_found, matches_expected)

    # Check if the plots have been generated
    assert os.path.exists(
        rascil_path(f"test_results/test_imaging_qa_{tag}_restored_plot.png")
    )
    assert os.path.exists(
        rascil_path(f"test_results/test_imaging_qa_{tag}_sources_plot.png")
    )
    assert os.path.exists(
        rascil_path(f"test_results/test_imaging_qa_{tag}_background_plot.png")
    )
    assert os.path.exists(
        rascil_path(f"test_results/test_imaging_qa_{tag}_restored_power_spectrum.png")
    )

    if residual_file is not None:
        assert os.path.exists(
            rascil_path(
                f"test_results/test_imaging_qa_{tag}_residual_residual_hist.png"
            )
        )
        assert os.path.exists(
            rascil_path(
                f"test_results/test_imaging_qa_{tag}_residual_residual_power_spectrum.png"
            )
        )

    assert os.path.exists(
        rascil_path(f"test_results/test_imaging_qa_{tag}_position_value.png")
    )
    assert os.path.exists(
        rascil_path(f"test_results/test_imaging_qa_{tag}_position_error.png")
    )
    assert os.path.exists(
        rascil_path(f"test_results/test_imaging_qa_{tag}_position_distance.png")
    )
    assert os.path.exists(
        rascil_path(f"test_results/test_imaging_qa_{tag}_flux_value.png")
    )
    assert os.path.exists(
        rascil_path(f"test_results/test_imaging_qa_{tag}_flux_ratio.png")
    )
    assert os.path.exists(
        rascil_path(f"test_results/test_imaging_qa_{tag}_flux_histogram.png")
    )

    assert os.path.exists(
        rascil_path(f"test_results/test_imaging_qa_{tag}_position_quiver.png")
    )

    assert os.path.exists(
        rascil_path(f"test_results/test_imaging_qa_{tag}_gaussian_beam_position.png")
    )

    if nchan > 1:
        assert os.path.exists(
            rascil_path(f"test_results/test_imaging_qa_{tag}_spec_index.png")
        )
        assert os.path.exists(
            rascil_path(
                f"test_results/test_imaging_qa_{tag}_spec_index_diagnostics_flux.png"
            )
        )
        assert os.path.exists(
            rascil_path(
                f"test_results/test_imaging_qa_{tag}_spec_index_diagnostics_dist.png"
            )
        )

    # test new csv file generated and accuracy
    csv_file = rascil_path(f"test_results/test_imaging_qa_{tag}_taylor1_corrected.csv")
    assert os.path.exists(csv_file)

    # This part does not work yet: skip for now
    # data = pd.read_csv(csv_file, engine="python")
    # indexes = data["Spectral index"]
    # numpy.testing.assert_array_almost_equal(indexes / spec_indx, 1.0, decimal=1)

    # test that create_index() generates the html and md files,
    # at the end of analyze_image()
    assert os.path.exists(rascil_path("test_results/index.html"))
    assert os.path.exists(rascil_path("test_results/index.md"))

    # clean up directory
    if persist is False:
        imglist = glob.glob(rascil_path(f"test_results/test_imaging_qa_{tag}*"))
        for f in imglist:
            os.remove(f)
        try:
            os.remove(rascil_path("test_results/index.html"))
            os.remove(rascil_path("test_results/index.md"))
        except OSError:
            pass
