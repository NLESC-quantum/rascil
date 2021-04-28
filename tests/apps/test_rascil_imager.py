""" Unit processing_components for rascil-imager

"""
import logging
import pytest
import shutil

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.apps.rascil_imager import cli_parser, imager
from rascil.data_models import SkyModel
from rascil.data_models.parameters import rascil_path
from rascil.data_models.polarisation import PolarisationFrame
from rascil.data_models.data_model_helpers import export_skymodel_to_hdf5
from rascil.processing_components import (
    export_blockvisibility_to_ms,
    concatenate_blockvisibility_frequency,
    find_skycomponents,
)
from rascil.processing_components.util.performance import (
    performance_store_dict, performance_environment
)
from rascil.processing_components import import_image_from_fits
from rascil.processing_components.calibration.operations import (
    create_gaintable_from_blockvisibility,
    apply_gaintable,
)
from rascil.processing_components.image.operations import (
    export_image_to_fits,
    qa_image,
    smooth_image,
)
from rascil.processing_components.imaging import dft_skycomponent_visibility
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.simulation import (
    ingest_unittest_visibility,
    create_unittest_model,
    create_unittest_components,
)
from rascil.processing_components.simulation import simulate_gaintable
from rascil.processing_components.skycomponent.operations import insert_skycomponent
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute

log = logging.getLogger("rascil-logger")
log.setLevel(logging.WARNING)
default_run = True

@pytest.mark.parametrize(
    "enabled, tag, use_dask, nmajor, mode, add_errors, flux_max, flux_min, component_threshold, component_method, offset",
    [
        (
            default_run,
            "invert",
            True,
            0,
            "invert",
            False,
            97.86001353082986,
            -13.658737888701651,
            None,
            None,
            5.0,
        ),
        (
            default_run,
            "invert_offset",
            True,
            0,
            "invert",
            False,
            94.73840939270013,
            -13.614616421597278,
            None,
            None,
            5.5,
        ),
        (
            default_run,
            "ical",
            True,
            9,
            "ical",
            True,
            115.71426187661086,
            -1.408050738866888,
            None,
            None,
            5.0,
        ),
        (
            default_run,
            "cip",
            True,
            9,
            "cip",
            False,
            116.74934844229546,
            -0.2961349090496892,
            None,
            "None",
            5.0,
        ),
        (
            default_run,
            "cip_offset",
            True,
            9,
            "cip",
            False,
            107.80020118248238,
            -0.6062812476137865,
            None,
            "None",
            5.5,
        ),
        (
            default_run,
            "cip_offset_fit",
            True,
            9,
            "cip",
            False,
            111.84507020640052,
            -1.421622304601026,
            "10",
            "fit",
            5.5,
        ),
    ],
)
def test_rascil_imager(
    enabled,
    tag,
    use_dask,
    nmajor,
    mode,
    add_errors,
    flux_max,
    flux_min,
    component_threshold,
    component_method,
    offset,
):
    """

    :param enabled: Turn this test on?
    :param tag: Tag for files generated
    :param use_dask: Use dask for processing. Set to False for debugging
    :param nmajor: Number of CLEAN major cycles
    :param mode: rqscil imager mode: invert or cip or ical
    :param add_errors: Add calibration errors (needed for ical testing)
    :param flux_max: Maximum flux in result (tested to 1e-7)
    :param flux_min: Minimum flux in result (tested to 1e-7)
    :param component_threshold: Flux above which components are searched and fitted in first deconvolution
    :param component_method: Method to find components: fit or None
    :param offset: Offset of test pattern in RA pizels
    :return:
    """

    if not enabled:
        return True

    nfreqwin = 7
    dospectral = True
    zerow = False
    dopol = False
    persist = True

    # We always want the same numbers
    from numpy.random import default_rng

    rng = default_rng(1805550721)

    rsexecute.set_client(use_dask=use_dask)

    npixel = 512
    low = create_named_configuration("LOWBD2", rmax=750.0)
    freqwin = nfreqwin
    ntimes = 3
    times = numpy.linspace(-3.0, +3.0, ntimes) * numpy.pi / 12.0
    frequency = numpy.linspace(0.8e8, 1.2e8, freqwin)

    if freqwin > 1:
        channelwidth = numpy.array(freqwin * [frequency[1] - frequency[0]])
    else:
        channelwidth = numpy.array([1e6])

    if dopol:
        vis_pol = PolarisationFrame("linear")
        image_pol = PolarisationFrame("stokesIQUV")
        f = numpy.array([100.0, 20.0, 0.0, 0.0])
    else:
        vis_pol = PolarisationFrame("stokesI")
        image_pol = PolarisationFrame("stokesI")
        f = numpy.array([100.0])

    if dospectral:
        flux = numpy.array([f * numpy.power(freq / 1e8, -0.7) for freq in frequency])
    else:
        flux = numpy.array([f])

    phasecentre = SkyCoord(
        ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
    )
    bvis_list = [
        rsexecute.execute(ingest_unittest_visibility, nout=1)(
            low,
            [frequency[i]],
            [channelwidth[i]],
            times,
            vis_pol,
            phasecentre,
            zerow=zerow,
        )
        for i in range(nfreqwin)
    ]
    bvis_list = rsexecute.persist(bvis_list)

    model_imagelist = [
        rsexecute.execute(create_unittest_model, nout=1)(
            bvis_list[i], image_pol, npixel=npixel, cellsize=0.0005
        )
        for i in range(nfreqwin)
    ]
    model_imagelist = rsexecute.persist(model_imagelist)

    components_list = [
        rsexecute.execute(create_unittest_components)(
            model_imagelist[freqwin],
            flux[freqwin, :][numpy.newaxis, :],
            offset=(offset, 0.0),
        )
        for freqwin, m in enumerate(model_imagelist)
    ]
    components_list = rsexecute.persist(components_list)

    bvis_list = [
        rsexecute.execute(dft_skycomponent_visibility)(
            bvis_list[freqwin], components_list[freqwin]
        )
        for freqwin, _ in enumerate(bvis_list)
    ]
    bvis_list = rsexecute.persist(bvis_list)

    if persist:
        components_list = rsexecute.compute(components_list, sync=True)

        model_imagelist = [
            rsexecute.execute(insert_skycomponent, nout=1)(
                model_imagelist[freqwin], components_list[freqwin]
            )
            for freqwin in range(nfreqwin)
        ]

        model_imagelist = rsexecute.compute(model_imagelist, sync=True)

        model = model_imagelist[0]
        cmodel = smooth_image(model)
        export_image_to_fits(
            model, rascil_path("test_results/test_rascil_imager_model.fits")
        )
        export_image_to_fits(
            cmodel, rascil_path("test_results/test_rascil_imager_cmodel.fits")
        )
        found_components = find_skycomponents(cmodel)
        sm = SkyModel(components=components_list[3])
        export_skymodel_to_hdf5(
            sm, rascil_path("test_results/test_rascil_imager_cmodel_original.hdf")
        )
        sm = SkyModel(components=found_components)
        export_skymodel_to_hdf5(
            sm, rascil_path("test_results/test_rascil_imager_cmodel_found.hdf")
        )

    if add_errors:
        seeds = [rng.integers(low=1, high=2 ** 32 - 1) for i in range(nfreqwin)]
        if nfreqwin == 5:
            assert seeds == [
                3822708302,
                2154889844,
                3073218956,
                3754981936,
                3778183766,
            ], seeds

        def sim_and_apply(vis, seed):
            gt = create_gaintable_from_blockvisibility(vis)
            gt = simulate_gaintable(
                gt,
                phase_error=0.1,
                amplitude_error=0.0,
                smooth_channels=1,
                leakage=0.0,
                seed=seed,
            )
            return apply_gaintable(vis, gt)

        # Do this without Dask since the random number generation seems to go wrong
        bvis_list = [
            rsexecute.execute(sim_and_apply)(bvis_list[i], seeds[i])
            for i in range(freqwin)
        ]
        bvis_list = rsexecute.compute(bvis_list, sync=True)
        bvis_list = rsexecute.scatter(bvis_list)

    shutil.rmtree(
        rascil_path(f"test_results/test_rascil_imager_{tag}.ms"), ignore_errors=True
    )
    bvis_list = rsexecute.compute(bvis_list, sync=True)
    bvis_list = [concatenate_blockvisibility_frequency(bvis_list)]
    export_blockvisibility_to_ms(
        rascil_path(f"test_results/test_rascil_imager_{tag}.ms"), bvis_list
    )

    rsexecute.close()

    invert_args = [
        "--mode",
        f"{mode}",
        "--use_dask",
        f"{use_dask}",
        "--performance_file",
        rascil_path(f"test_results/test_rascil_imager_performance_{tag}.json"),
        "--ingest_msname",
        rascil_path(f"test_results/test_rascil_imager_{tag}.ms"),
        "--ingest_vis_nchan",
        f"{nfreqwin}",
        "--ingest_dd",
        "0",
        "--ingest_chan_per_blockvis",
        "1",
        "--imaging_npixel",
        "512",
        "--imaging_cellsize",
        "0.0005",
        "--imaging_dft_kernel",
        "cpu_looped",
    ]

    clean_args = [
        "--clean_nmajor",
        f"{nmajor}",
        "--clean_niter",
        "1000",
        "--clean_algorithm",
        "mmclean",
        "--clean_nmoment",
        "2",
        "--clean_gain",
        "0.1",
        "--clean_scales",
        "0",
        "--clean_threshold",
        "0.003",
        "--clean_fractional_threshold",
        "0.03",
        "--clean_facets",
        "1",
        "--clean_restored_output",
        "list",
        "--clean_restore_facets",
        "4",
        "--clean_restore_overlap",
        "8"
    ]
    if component_threshold is not None and component_method is not None:
        clean_args += [
            "--clean_component_threshold",
            f"{component_threshold}",
            "--clean_component_method",
            f"{component_method}",
        ]
    else:
        clean_args += [
            "--clean_component_threshold",
            "1e15",
            "--clean_component_method",
            "fit",
        ]

    calibration_args = [
        "--calibration_T_first_selfcal",
        "2",
        "--calibration_T_phase_only",
        "True",
        "--calibration_T_timeslice",
        "0.0",
        "--calibration_G_first_selfcal",
        "5",
        "--calibration_G_phase_only",
        "False",
        "--calibration_G_timeslice",
        "1200.0",
        "--calibration_B_first_selfcal",
        "8",
        "--calibration_B_phase_only",
        "False",
        "--calibration_B_timeslice",
        "1.0e5",
        "--calibration_global_solution",
        "True",
        "--calibration_context",
        "TG",
    ]

    parser = cli_parser()
    if mode == "invert":
        args = parser.parse_args(invert_args)
    elif mode == "cip":
        args = parser.parse_args(invert_args + clean_args)
    elif mode == "ical":
        args = parser.parse_args(invert_args + clean_args + calibration_args)
    else:
        return ValueError(f"rascil-imager: Unknown mode {mode}")

    performance_environment(args.performance_file, mode="w")
    performance_store_dict(args.performance_file, "cli_args", vars(args), mode="a")

    if mode == "invert":
        dirtyname = imager(args)
        dirty = import_image_from_fits(dirtyname)
        qa = qa_image(dirty)
    elif mode == "cip":
        restoredname = imager(args)[2]
        dirty = import_image_from_fits(restoredname)
        qa = qa_image(dirty)
    elif mode == "ical":
        restoredname = imager(args)[2]
        dirty = import_image_from_fits(restoredname)
        qa = qa_image(dirty)
    else:
        return ValueError(f"rascil-imager: Unknown mode {mode}")

    numpy.testing.assert_allclose(qa.data["max"], flux_max, atol=1e-7, err_msg=f"{qa}")
    numpy.testing.assert_allclose(qa.data["min"], flux_min, atol=1e-7, err_msg=f"{qa}")
