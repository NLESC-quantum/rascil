""" Unit tests for pipelines expressed via rsexecute
"""

import functools
import os
import pprint

import numpy
import unittest
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models import export_skymodel_to_hdf5, export_skycomponent_to_hdf5

# These are the RASCIL functions we need
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import (
    export_image_to_fits,
    qa_image,
    create_low_test_skymodel_from_gleam,
    create_low_test_skycomponents_from_gleam,
    create_image_from_visibility,
    image_gather_channels,
    create_low_test_beam,
    calculate_blockvisibility_azel,
)
from rascil.workflows import (
    predict_skymodel_list_rsexecute_workflow,
    simulate_list_rsexecute_workflow,
    weight_list_rsexecute_workflow,
)
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute
from rascil.workflows.rsexecute.pipelines.pipeline_skymodel_rsexecute import (
    continuum_imaging_skymodel_list_rsexecute_workflow,
)

pp = pprint.PrettyPrinter()

import logging

log = logging.getLogger("rascil-logger")

logging.info("Starting imaging-pipeline")
log.setLevel(logging.WARNING)

# These tests probe whether the results depend on whether Dask is used and also whether
# optimisation in Dask is used.

default_run = False


@unittest.skip("Too expensive for CI/CD")
@pytest.mark.parametrize(
    "default_run, use_dask, optimise, test_max, test_min, sensitivity, tag, rmax",
    [
        (
            not default_run,
            True,
            True,
            6.787887014253024,
            -0.08546184846602671,
            False,
            "Dask_Optimise_300",
            300.0,
        ),
        (
            default_run,
            True,
            True,
            6.787887014253024,
            -0.08546184846602671,
            False,
            "Dask_Optimise_300",
            600.0,
        ),
        (
            default_run,
            True,
            True,
            6.6780296123876735,
            -0.056910264450297565,
            True,
            "Dask_Optimise_Sensitivity_600",
            600.0,
        ),
        (
            default_run,
            True,
            True,
            6.6780296123876735,
            -0.056910264450297565,
            True,
            "Dask_Optimise_Sensitivity_1200",
            1200.0,
        ),
        (
            default_run,
            True,
            False,
            6.787887014253024,
            -0.08546184846602671,
            False,
            "Dask_No_Optimise_600",
            600.0,
        ),
        (
            default_run,
            False,
            False,
            6.787887014253024,
            -0.08546184846602671,
            False,
            "No_Dask_600",
            600.0,
        ),
    ],
)
def test_imaging_pipeline(
    default_run, use_dask, optimise, test_max, test_min, sensitivity, tag, rmax
):
    """Test of imaging pipeline

    :param default_run: Run this?
    :param use_dask: - Use dask for processing
    :param optimise: - Enable dask graph optimisation
    :param test_max, test_min: max, min in tests
    :param flat_sky: Normalise out the primary beam?
    :param tag: Information tag for file names
    """

    if not default_run:
        return

    rsexecute.set_client(use_dask=use_dask, optim=optimise)

    from rascil.data_models.parameters import rascil_path

    dir = rascil_path("test_results")

    persist = os.getenv("RASCIL_PERSIST", False)

    nfreqwin = 9
    ntimes = 5
    frequency = numpy.linspace(0.8e8, 1.2e8, nfreqwin)
    if nfreqwin > 1:
        channel_bandwidth = numpy.array(nfreqwin * [frequency[1] - frequency[0]])
    else:
        channel_bandwidth = numpy.array([2e7])
    times = numpy.linspace(-numpy.pi / 3.0, numpy.pi / 3.0, ntimes)
    phasecentre = SkyCoord(
        ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
    )

    bvis_list = simulate_list_rsexecute_workflow(
        "LOWBD2",
        frequency=frequency,
        channel_bandwidth=channel_bandwidth,
        times=times,
        phasecentre=phasecentre,
        order="frequency",
        rmax=rmax,
        format="blockvis",
        zerow=False,
    )
    bvis_list = rsexecute.persist(bvis_list)

    if rmax > 600.0:
        npixel = 2048
    elif rmax > 300.0:
        npixel = 1024
    else:
        npixel = 512
    cellsize = 5e-4 * 600.0 / rmax

    model_list = [
        rsexecute.execute(create_image_from_visibility, nout=1)(
            bvis_list[f],
            npixel=npixel,
            frequency=[frequency[f]],
            channel_bandwidth=[channel_bandwidth[f]],
            cellsize=cellsize,
            phasecentre=phasecentre,
            polarisation_frame=PolarisationFrame("stokesI"),
            chunksize=None,
        )
        for f, freq in enumerate(frequency)
    ]
    model_list = rsexecute.persist(model_list)
    bvis_list = weight_list_rsexecute_workflow(bvis_list, model_imagelist=model_list)

    gleam_skymodel_list = [
        rsexecute.execute(create_low_test_skymodel_from_gleam, nout=1)(
            npixel=npixel,
            frequency=[frequency[f]],
            channel_bandwidth=[channel_bandwidth[f]],
            cellsize=cellsize,
            phasecentre=phasecentre,
            polarisation_frame=PolarisationFrame("stokesI"),
            flux_limit=1.0,
        )
        for f, freq in enumerate(frequency)
    ]
    gleam_skymodel_list = rsexecute.persist(gleam_skymodel_list)
    log.info("About to make GLEAM model")

    gleam_components = create_low_test_skycomponents_from_gleam(
        frequency=frequency,
        polarisation_frame=PolarisationFrame("stokesI"),
        phasecentre=phasecentre,
        flux_limit=1.0,
        radius=0.1,
    )

    if sensitivity:

        def create(vis, model):
            az, el = calculate_blockvisibility_azel(vis)
            az = numpy.mean(az)
            el = numpy.mean(el)
            return create_low_test_beam(model, use_local=False, azel=(az, el))

        get_pb = functools.partial(create)
    else:
        get_pb = None

    predicted_vislist = predict_skymodel_list_rsexecute_workflow(
        bvis_list,
        skymodel_list=gleam_skymodel_list,
        context="ng",
        get_pb=get_pb,
    )
    predicted_vislist = rsexecute.persist(predicted_vislist)

    continuum_imaging_list = continuum_imaging_skymodel_list_rsexecute_workflow(
        predicted_vislist,
        model_imagelist=model_list,
        skymodel_list=None,
        get_pb=get_pb,
        context="ng",
        algorithm="mmclean",
        scales=[0],
        niter=1000,
        fractional_threshold=0.1,
        threshold=0.01,
        nmoment=3,
        nmajor=10,
        gain=0.7,
        deconvolve_facets=1,
        deconvolve_overlap=32,
        deconvolve_taper="tukey",
        psf_support=64,
        do_wstacking=True,
        flat_sky=False,
    )

    continuum_imaging_list = rsexecute.compute(continuum_imaging_list, sync=True)

    skymodel_list = continuum_imaging_list[2]
    export_skycomponent_to_hdf5(
        gleam_components,
        "%s/test-continuum_imaging_%s_components.hdf" % (dir, tag),
    )
    export_skymodel_to_hdf5(
        skymodel_list,
        "%s/test-continuum_imaging_%s_skymodel.hdf" % (dir, tag),
    )
    # Write frequency cubes
    deconvolved = image_gather_channels(
        [skymodel_list[chan].image for chan in range(nfreqwin)]
    )
    residual = image_gather_channels(
        [continuum_imaging_list[0][chan][0] for chan in range(nfreqwin)]
    )
    restored = image_gather_channels(
        [continuum_imaging_list[1][chan] for chan in range(nfreqwin)]
    )

    log.info(qa_image(deconvolved, context="Clean image "))
    export_image_to_fits(
        deconvolved,
        "%s/test-continuum_imaging_%s_clean.fits" % (dir, tag),
    )
    log.info(qa_image(residual, context="Residual clean image "))
    export_image_to_fits(
        residual,
        "%s/test-continuum_imaging_%s_residual.fits" % (dir, tag),
    )

    if sensitivity:
        sens = image_gather_channels(
            [continuum_imaging_list[0][chan][1] for chan in range(nfreqwin)]
        )
        log.info(qa_image(sens, context="Sensitivity image "))
        export_image_to_fits(
            sens,
            "%s/test-continuum_imaging_%s_sensitivity.fits" % (dir, tag),
        )

    qa = qa_image(restored, context="Restored image ")
    log.info(qa)
    export_image_to_fits(
        restored,
        "%s/test-continuum_imaging_%s_restored.fits" % (dir, tag),
    )

    # Correct values for no skycomponent extraction
    assert abs(qa.data["max"] - test_max) < 1e-7, str(qa)
    assert abs(qa.data["min"] - test_min) < 1e-7, str(qa)
