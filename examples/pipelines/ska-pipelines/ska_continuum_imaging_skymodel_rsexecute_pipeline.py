"""
Continuum processing pipeline
"""
# # Pipeline processing using Dask

from rascil.data_models.parameters import rascil_path

results_dir = "./results/"

from rascil.data_models import PolarisationFrame

from rascil.processing_components import (
    export_image_to_fits,
    qa_image,
    create_image_from_visibility,
    create_blockvisibility_from_ms,
    image_gather_channels,
)

from rascil.workflows import (
    continuum_imaging_skymodel_list_rsexecute_workflow,
    weight_list_rsexecute_workflow,
)

from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute

import logging


def init_logging():
    logging.basicConfig(
        filename="%s/ska-pipeline.log" % results_dir,
        filemode="a",
        format="%(asctime)s.%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )


if __name__ == "__main__":
    log = logging.getLogger("rascil-logger")
    logging.info("Starting continuum imaging pipeline")

    rsexecute.set_client(use_dask=True)
    print(rsexecute.client)
    rsexecute.run(init_logging)

    nfreqwin = 8

    # Load data from previous simulation
    vis_list = [
        rsexecute.execute(create_blockvisibility_from_ms)(
            r"%s/ska-pipeline_simulation_vislist_%d.ms" % (results_dir, v)
        )[0]
        for v in range(nfreqwin)
    ]

    vis_list = rsexecute.persist(vis_list)

    cellsize = 0.0003
    npixel = 1024
    pol_frame = PolarisationFrame("stokesI")

    model_list = [
        rsexecute.execute(create_image_from_visibility)(
            v, npixel=npixel, cellsize=cellsize, polarisation_frame=pol_frame
        )
        for v in vis_list
    ]

    model_list = rsexecute.persist(model_list)

    vis_list = weight_list_rsexecute_workflow(vis_list, model_list)

    imaging_context = "ng"

    continuum_imaging_list = continuum_imaging_skymodel_list_rsexecute_workflow(
        vis_list,
        model_imagelist=model_list,
        context=imaging_context,
        scales=[0],
        algorithm="mmclean",
        nmoment=2,
        niter=1000,
        fractional_threshold=0.3,
        threshold=0.1,
        nmajor=5,
        gain=0.25,
        deconvolve_facets=1,
        deconvolve_overlap=0,
        timeslice="auto",
        psf_support=128,
        restored_output="list",
    )

    log.info("About to run full graph")
    result = rsexecute.compute(continuum_imaging_list, sync=True)
    rsexecute.close()

    # The return is:
    #    (nchan * (residual_image, sumwt), nchan * restored_image, nchan * skymodel)
    residual = image_gather_channels([result[0][chan][0] for chan in range(nfreqwin)])
    restored = image_gather_channels([result[1][chan] for chan in range(nfreqwin)])
    deconvolved = image_gather_channels(
        [result[2][chan].image for chan in range(nfreqwin)]
    )

    log.info(qa_image(deconvolved, context="Clean image cube"))
    export_image_to_fits(
        deconvolved,
        "%s/ska-continuum-imaging_rsexecute_deconvolved_cube.fits" % (results_dir),
    )

    log.info(qa_image(restored, context="Restored clean image cube"))
    export_image_to_fits(
        restored,
        "%s/ska-continuum-imaging_rsexecute_restored_cube.fits" % (results_dir),
    )

    log.info(qa_image(residual, context="Residual clean image cube"))
    export_image_to_fits(
        residual,
        "%s/ska-continuum-imaging_rsexecute_residual_cube.fits" % (results_dir),
    )
