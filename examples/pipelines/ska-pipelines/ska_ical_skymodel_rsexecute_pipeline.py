"""
ICAL pipeline
"""

results_dir = "./"

from rascil.data_models import PolarisationFrame, export_gaintable_to_hdf5


from rascil.processing_components import (
    export_image_to_fits,
    qa_image,
    qa_gaintable,
    create_image_from_visibility,
    create_blockvisibility_from_ms,
    image_gather_channels,
    create_calibration_controls,
)

from rascil.workflows import (
    ical_skymodel_list_rsexecute_workflow,
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
    logging.info("Starting ICAL pipeline")

    rsexecute.set_client(use_dask=True)
    print(rsexecute.client)
    rsexecute.run(init_logging)

    nfreqwin = 8

    # Load data from previous simulation
    vis_list = [
        rsexecute.execute(create_blockvisibility_from_ms)(
            "%s/ska-pipeline_simulation_%d.ms" % (results_dir, v)
        )[0]
        for v in range(nfreqwin)
    ]

    vis_list = rsexecute.persist(vis_list)

    # Define image properties
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

    # Weight the data
    vis_list = weight_list_rsexecute_workflow(vis_list, model_list)

    imaging_context = "ng"

    # The calibration sequence is defined in a dictionary. A default
    # version is constructed using create_calibration_controls()
    controls = create_calibration_controls()
    controls["T"]["first_selfcal"] = 1
    controls["T"]["phase_only"] = True
    controls["T"]["timeslice"] = "auto"

    # Now we can define the graph for ICAL
    ical_list = ical_skymodel_list_rsexecute_workflow(
        vis_list,
        model_imagelist=model_list,
        context=imaging_context,
        scales=[0],
        algorithm="mmclean",
        nmoment=2,
        niter=1000,
        fractional_threshold=0.3,
        threshold=0.02,
        nmajor=5,
        gain=0.25,
        deconvolve_facets=1,
        deconvolve_overlap=0,
        timeslice="auto",
        psf_support=128,
        controls=controls,
        global_solution=False,
        calibration_context="T",
        do_selfcal=True,
        restored_output="list",
    )

    log.info("About to run ICAL workflow")
    result = rsexecute.compute(ical_list, sync=True)
    rsexecute.close()

    # The result is:
    #    (nchan * (residual_image, sumwt), nchan * restored_image, nchan * skymodel)
    residual = image_gather_channels([result[0][chan][0] for chan in range(nfreqwin)])
    restored = image_gather_channels([result[1][chan] for chan in range(nfreqwin)])
    deconvolved = image_gather_channels(
        [result[2][chan].image for chan in range(nfreqwin)]
    )
    gt_list = result[3]

    log.info(qa_image(deconvolved, context="Clean image cube"))
    export_image_to_fits(
        deconvolved, "%s/ska-ical_rsexecute_deconvolved_cube.fits" % (results_dir)
    )

    log.info(qa_image(restored, context="Restored clean image cube"))
    export_image_to_fits(
        restored, "%s/ska-ical_rsexecute_restored_cube.fits" % (results_dir)
    )

    log.info(qa_image(residual, context="Residual clean image cube"))
    export_image_to_fits(
        residual, "%s/ska-ical_rsexecute_residual_cube.fits" % (results_dir)
    )

    # The gaintable are nchan*{"T":gt}
    log.info(qa_gaintable(gt_list[0]["T"]))
    agt_list = [gt_list[chan]["T"] for chan in range(nfreqwin)]
    export_gaintable_to_hdf5(
        agt_list, "%s/ska-ical_rsexecute_gaintable.hdf" % (results_dir)
    )
