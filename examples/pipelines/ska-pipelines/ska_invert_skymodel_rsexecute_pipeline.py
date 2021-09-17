"""
Imaging pipeline
"""

# # Pipeline processing using Dask

from rascil.data_models.parameters import rascil_path

results_dir = "./results/"

from rascil.data_models import PolarisationFrame, SkyModel

from rascil.processing_components import (
    export_image_to_fits,
    qa_image,
    create_image_from_visibility,
    create_blockvisibility_from_ms,
    image_gather_channels,
)

from rascil.workflows import (
    invert_skymodel_list_rsexecute_workflow,
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
    logging.info("Starting Imaging pipeline")

    rsexecute.set_client(use_dask=True)
    log.info(rsexecute.client)
    rsexecute.run(init_logging)

    nfreqwin = 8

    # Load data from previous simulation
    vis_list = [
        rsexecute.execute(create_blockvisibility_from_ms)(
            "%s/ska-pipeline_simulation_vislist_%d.ms" % (results_dir, v)
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

    skymodel_list = [
        rsexecute.execute(SkyModel)(components=[], image=model) for model in model_list
    ]

    imaging_context = "ng"

    dirty_list = invert_skymodel_list_rsexecute_workflow(
        vis_list, skymodel_list=skymodel_list, context=imaging_context
    )

    log.info("About to run full graph")
    result = rsexecute.compute(dirty_list, sync=True)
    centre = nfreqwin // 2
    dirty, sumwt = result[centre]

    rsexecute.close()

    export_image_to_fits(
        dirty, "%s/ska-imaging_rsexecute_dirty_centre.fits" % (results_dir)
    )

    dirty_cube = image_gather_channels([r[0] for r in result])
    log.info(qa_image(dirty_cube, context="Dirty image cube"))
    export_image_to_fits(
        dirty_cube, "%s/ska-imaging_rsexecute_dirty_cube.fits" % (results_dir)
    )

    exit(0)
