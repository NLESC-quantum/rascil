""" Unit tests for pipelines expressed via rsexecute
"""

import os
import pprint

import numpy
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from rascil.data_models import export_skymodel_to_hdf5, export_skycomponent_to_hdf5

# These are the RASCIL functions we need
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import (
    show_image,
    export_image_to_fits,
    qa_image,
    create_low_test_image_from_gleam,
    create_low_test_skycomponents_from_gleam,
    create_image_from_visibility,
    image_gather_channels
)
from rascil.workflows import (
    predict_list_rsexecute_workflow,
    simulate_list_rsexecute_workflow,
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


@pytest.mark.parametrize(
    "use_dask, optimise, test_max, test_min",
    [
        (True, True, 4.094169571405569, -0.005846355119035163),
#        (True, False, 4.094169571405569, -0.005846355119035163),
#        (False, False, 4.094169571405569, -0.005846355119035163),
    ],
)
def test_imaging_pipeline(use_dask, optimise, test_max, test_min):
    """Test of imaging pipeline

    :param use_dask: - Use dask for processing
    :param optimise: - Enable dask graph optimisation
    :param component_method: Method to find bright components pixels or fit
    :param component_threshold: - Threshold in Jy/pixel for classifying as component
    :param test_max, test_min:: max, min in tests."""
    rsexecute.set_client(use_dask=use_dask, optim=optimise)

    from rascil.data_models.parameters import rascil_path

    dir = rascil_path("test_results")

    persist = os.getenv("RASCIL_PERSIST", False)

    nfreqwin = 5
    ntimes = 5
    rmax = 300.0
    frequency = numpy.linspace(1e8, 1.2e8, nfreqwin)
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

    npixel = 512
    cellsize = 1e-3

    gleam_model = [
        rsexecute.execute(create_low_test_image_from_gleam, nout=1)(
            npixel=npixel,
            frequency=[frequency[f]],
            channel_bandwidth=[channel_bandwidth[f]],
            cellsize=cellsize,
            phasecentre=phasecentre,
            polarisation_frame=PolarisationFrame("stokesI"),
            flux_limit=1.0,
            applybeam=True,
        )
        for f, freq in enumerate(frequency)
    ]
    log.info("About to make GLEAM model")
    
    gleam_components = create_low_test_skycomponents_from_gleam(
            frequency=frequency,
            polarisation_frame=PolarisationFrame("stokesI"),
        phasecentre=phasecentre,
            flux_limit=1.0,
        radius=0.1
    )


    predicted_vislist = predict_list_rsexecute_workflow(
        bvis_list, gleam_model, context="ng"
    )

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

    continuum_imaging_list = continuum_imaging_skymodel_list_rsexecute_workflow(
        predicted_vislist,
        model_imagelist=model_list,
        skymodel_list=None,
        context="ng",
        algorithm="mmclean",
        scales=[0],
        niter=100,
        fractional_threshold=0.1,
        threshold=0.01,
        nmoment=1,
        nmajor=5,
        gain=0.7,
        deconvolve_facets=4,
        deconvolve_overlap=32,
        deconvolve_taper="tukey",
        psf_support=64,
        do_wstacking=True,
    )

    centre = nfreqwin // 2
    continuum_imaging_list = rsexecute.compute(continuum_imaging_list, sync=True)
    deconvolved = continuum_imaging_list[0][centre]
    residual = continuum_imaging_list[1][centre]
    restored_cube = image_gather_channels(continuum_imaging_list[2])

    restored_plane = continuum_imaging_list[2][centre]
    skymodel_list = continuum_imaging_list[3]

    export_skycomponent_to_hdf5(
        gleam_components,
        "%s/test-imaging-pipeline-dask_continuum_imaging_components.hdf" % (dir),
    )

    export_skymodel_to_hdf5(
        skymodel_list,
        "%s/test-imaging-pipeline-dask_continuum_imaging_skymodel.hdf" % (dir),
    )

    f = show_image(
        deconvolved, title="Clean image - no selfcal", cm="Greys", vmax=0.1, vmin=-0.01
    )
    log.info(qa_image(deconvolved, context="Clean image - no selfcal"))
    export_image_to_fits(
        deconvolved,
        "%s/test-imaging-pipeline-dask_continuum_imaging_clean.fits" % (dir),
    )

    plt.show()

    f = show_image(residual[0], title="Residual clean image - no selfcal", cm="Greys")
    log.info(qa_image(residual[0], context="Residual clean image - no selfcal"))
    plt.show()
    export_image_to_fits(
        residual[0],
        "%s/test-imaging-pipeline-dask_continuum_imaging_residual.fits" % (dir),
    )

    f = show_image(
        restored_plane,
        title="Restored clean image - no selfcal",
        cm="Greys",
        vmax=1.0,
        vmin=-0.1,
    )
    log.info(qa_image(restored_plane, context="Restored clean image - no selfcal"))
    plt.show()
    export_image_to_fits(
        restored_plane,
        "%s/test-imaging-pipeline-dask_continuum_imaging_restored.fits" % (dir),
    )

    qa = qa_image(restored_plane, context="Restored clean image - no selfcal")

    # Correct values for no skycomponent extraction
    assert abs(qa.data["max"] - test_max) < 1e-7, str(qa)
    assert abs(qa.data["min"] - test_min) < 1e-7, str(qa)


    export_image_to_fits(
        restored_cube,
        "%s/test-imaging-pipeline-dask_continuum_imaging_restored_cube.fits" % (dir),
    )
