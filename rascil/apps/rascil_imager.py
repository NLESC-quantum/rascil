""" RASCIL Continuum Imaging

"""

import argparse
import datetime
import logging
import os
import pprint
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from distributed import Client

from rascil.data_models import PolarisationFrame, export_skymodel_to_hdf5
from rascil.processing_components import (
    create_image_from_visibility,
    qa_image,
    show_image,
    convert_blockvisibility_to_stokesI,
    export_image_to_fits,
    image_gather_channels,
    create_calibration_controls,
)

from rascil.workflows import (
    weight_list_rsexecute_workflow,
    continuum_imaging_skymodel_list_rsexecute_workflow,
    sum_invert_results,
    create_blockvisibility_from_ms_rsexecute,
    ical_skymodel_list_rsexecute_workflow,
    invert_list_rsexecute_workflow,
    sum_invert_results_rsexecute,
)

from rascil.workflows.rsexecute.execution_support.rsexecute import (
    rsexecute,
    get_dask_client,
)

from rascil.apps.apps_parser import (
    apps_parser_imaging,
    apps_parser_cleaning,
    apps_parser_dask,
    apps_parser_ingest,
    apps_parser_app,
    apps_parser_calibration,
)

log = logging.getLogger("rascil-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


def cli_parser():
    """Get a command line parser and populate it with arguments

    First a CLI argument parser is created. Each function call adds more arguments to the parser.

    :return: CLI parser argparse
    """
    
    parser = argparse.ArgumentParser(description="RASCIL continuum imager",
                                     fromfile_prefix_chars="@")
    parser = apps_parser_app(parser)
    parser = apps_parser_ingest(parser)
    parser = apps_parser_imaging(parser)
    parser = apps_parser_calibration(parser)
    parser = apps_parser_cleaning(parser)
    parser = apps_parser_dask(parser)
    
    return parser


def imager(args):
    """Continuum imager

    The return contains names of the files written to disk as fits files.

    mode=invert: dirty image
    mode=cip: deconvolved image, residual image, restored image.
    mode=ical: deconvolved image, residual image, restored image

    :param args: argparse with appropriate arguments
    :return: Names of outputs as fits files
    """
    
    # We need to tell all the Dask workers to use the same log
    cwd = os.getcwd()
    
    assert args.ingest_msname is not None, "Input msname must be specified"
    
    if args.logfile is None:
        logfile = args.ingest_msname.replace(".ms", ".log")
    else:
        logfile = args.logfile
    
    def init_logging():
        logging.basicConfig(
            filename=logfile,
            filemode="a",
            format="%(asctime)s.%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%d/%m/%Y %I:%M:%S %p",
            level=logging.INFO,
        )
    
    init_logging()
    
    log.info("\nRASCIL Continuum Imager\n")
    
    starttime = datetime.datetime.now()
    log.info("Started : {}".format(starttime))
    log.info("Writing log to {}".format(logfile))
    
    # We can run distributed (use_dask=True) or in serial (use_dask=False). Using Dask is usually recommended
    if args.use_dask == "True":
        if args.dask_scheduler is not None:
            log.info("Using scheduler {}".format(args.dask_scheduler))
            client = Client(scheduler=args.dask_scheduler)
        else:
            client = get_dask_client(
                n_workers=args.dask_nworkers,
                threads_per_worker=args.dask_nthreads,
                memory_limit=args.dask_memory,
            )
        rsexecute.set_client(use_dask=True, client=client)
        rsexecute.init_statistics()
    else:
        rsexecute.set_client(use_dask=False)
    
    rsexecute.run(init_logging)
    
    log.info(pprint.pformat(vars(args)))
    
    log.info("Current working directory is {}".format(cwd))
    
    # Read in the MS into a list of BlockVisibility's
    # We start with an MS with e.g. 4 data_descriptors, each of which has e.g. 64 channels.
    # We average each dd over e.g. 2 blocks of e.g. 32 channels, giving e.g. 8 separate
    # BlockVisibility's
    
    # rsexecute is a slightly wrapped version of Dask. rsexecute.execute is
    # essentially the same as dask.delayed
    
    # Create a graph to read the MS into RASCIL BlockVisibility objects
    msname = args.ingest_msname
    
    dds = args.ingest_dd
    channels_per_dd = args.ingest_vis_nchan
    nchan_per_blockvis = args.ingest_chan_per_blockvis
    nout = channels_per_dd // nchan_per_blockvis
    
    bvis_list = create_blockvisibility_from_ms_rsexecute(
        msname=args.ingest_msname,
        dds=dds,
        nout=nout,
        nchan_per_blockvis=nchan_per_blockvis,
        average_channels=args.ingest_average_blockvis == "True",
    )
    bvis_list = rsexecute.persist(bvis_list)
    
    # If the cellsize has not been specified, we compute the blockvis now and
    # run the advisor
    cellsize = args.imaging_cellsize
    if cellsize is None:
        bvis_list = rsexecute.compute(bvis_list, sync=True)
        from rascil.processing_components import advise_wide_field
        
        advice = advise_wide_field(bvis_list[0], guard_band_image=3.0)
        cellsize = advice["cellsize"]
        log.info(f"Setting cellsize to {cellsize} rad")
    
    # Make only the Stokes I image so we convert the visibility to Stokes I
    if args.imaging_pol == "stokesI":
        bvis_list = [
            rsexecute.execute(convert_blockvisibility_to_stokesI)(bv)
            for bv in bvis_list
        ]
    
    npixel = args.imaging_npixel
    
    # Define the model to be used as a template, one for each BlockVisibility
    model_list = [
        rsexecute.execute(create_image_from_visibility)(
            bvis,
            npixel=npixel,
            nchan=args.imaging_nchan,
            cellsize=cellsize,
            polarisation_frame=PolarisationFrame(args.imaging_pol),
        )
        for bvis in bvis_list
    ]
    model_list = rsexecute.persist(model_list)
    
    # Create a graph to weight the data
    if args.imaging_weighting != "natural":
        bvis_list = weight_list_rsexecute_workflow(
            bvis_list,
            model_list,
            weighting=args.imaging_weighting,
            robustness=args.imaging_robustness,
        )
    bvis_list = rsexecute.persist(bvis_list)
    
    if args.mode == "cip":
        results = cip(args, bvis_list, model_list, msname)
    elif args.mode == "ical":
        results = ical(args, bvis_list, model_list, msname)
    elif args.mode == "invert":
        results = invert(args, bvis_list, model_list, msname)
    else:
        raise ValueError("Unknown mode {}".format(args.mode))
    
    # Save the processing statistics from Dask
    rsexecute.save_statistics(logfile.replace(".log", ""))
    rsexecute.close()
    
    log.info("Resulting image(s) {}".format(results))
    
    log.info("Started  : {}".format(starttime))
    log.info("Finished : {}".format(datetime.datetime.now()))
    
    return results


def cip(args, bvis_list, model_list, msname):
    """Run continuum imaging pipeline

    :param args: The parameters read from the CLI using argparse
    :param bvis_list: A list of or graph to make BlockVisibilitys
    :param model_list: A list of or graph to make model images
    :param msname: The filename of the MeasurementSet
    :return: Names of output images (deconvolved, residual, restored)
    """
    result = continuum_imaging_skymodel_list_rsexecute_workflow(
        bvis_list,  # List of BlockVisibilitys
        model_list,  # List of model images
        context=args.imaging_context,  # Use nifty-gridder
        threads=args.imaging_ng_threads,
        wstacking=args.imaging_w_stacking == "True",  # Correct for w term in gridding
        niter=args.clean_niter,  # iterations in minor cycle
        nmajor=args.clean_nmajor,  # Number of major cycles
        algorithm=args.clean_algorithm,
        gain=args.clean_gain,  # CLEAN loop gain
        scales=args.clean_scales,  # Scales for multi-scale cleaning
        fractional_threshold=args.clean_fractional_threshold,
        # Threshold per major cycle
        threshold=args.clean_threshold,  # Final stopping threshold
        nmoment=args.clean_nmoment,
        # Number of frequency moments (1 = no dependence)
        psf_support=args.clean_psf_support,
        # Support of PSF used in minor cycles (halfwidth in pixels)
        restored_output=args.clean_restored_output,  # Type of restored image
        deconvolve_facets=args.clean_facets,
        deconvolve_overlap=args.clean_overlap,
        deconvolve_taper=args.clean_taper,
        dft_compute_kernel=args.imaging_dft_kernel,
        component_threshold=args.clean_component_threshold,
        component_method=args.clean_component_method,
    )
    # Execute the Dask graph
    log.info("Starting compute of continuum imaging pipeline graph ")
    result = rsexecute.compute(result, sync=True)
    log.info("Finished compute of continuum imaging pipeline graph")
    
    imagename = msname.replace(".ms", "_nmoment{}_cip".format(args.clean_nmoment))
    return write_results(imagename, result)


def write_results(imagename, result):
    deconvolved, residual, restored, skymodel = result
    
    if isinstance(restored, list):
        # This is the case where we have a list of restored images
        restored = image_gather_channels(restored)
        log.info("Writing restored image as spectral cube")
        restoredname = imagename + "_restored_cube.fits"
        export_image_to_fits(restored, restoredname)
    else:
        log.info("Writing restored image as single plane at mid-frequency")
        restoredname = imagename + "_restored_centre.fits"
        export_image_to_fits(restored, restoredname)
    
    log.info(qa_image(restored, context="Restored"))
    show_image(restored, title=f"{imagename} Clean restored image")
    plt.savefig(imagename + "_restored.png")
    plt.show(block=False)

    # The residual images come as (image, weight) pairs so to get one image
    # we form a weight sum
    residual_image, sumwt = sum_invert_results(residual)
    log.info(qa_image(residual_image, context="Residual"))
    show_image(residual_image, title=f"{imagename} Clean residual image")
    plt.savefig(imagename + "_residual.png")
    plt.show(block=False)
    residualname = imagename + "_residual.fits"
    export_image_to_fits(residual_image, residualname)
    # The deconvolved image is a list of channels images. We gather these into
    # one image
    deconvolved_image = image_gather_channels(deconvolved)
    log.info(qa_image(deconvolved_image, context="Deconvolved"))
    show_image(deconvolved_image, title=f"{imagename} Clean deconvolved image")
    plt.savefig(imagename + "_deconvolved.png")
    plt.show(block=False)
    deconvolvedname = imagename + "_deconvolved.fits"
    export_image_to_fits(deconvolved_image, deconvolvedname)
    
    skymodelname = imagename + "_skymodel.hdf"
    export_skymodel_to_hdf5(skymodel, skymodelname)
    
    return (deconvolvedname, residualname, restoredname, skymodelname)


def ical(args, bvis_list, model_list, msname):
    """Run ICAL pipeline

    :param args: The parameters read from the CLI using argparse
    :param bvis_list: A list of or graph to make BlockVisibilitys
    :param model_list: A list of or graph to make model images
    :param msname: The filename of the MeasurementSet
    :return: Names of output images (deconvolved, residual, restored)
    """
    controls = create_calibration_controls()
    
    controls["T"]["first_selfcal"] = args.calibration_T_first_selfcal
    controls["T"]["phase_only"] = args.calibration_T_phase_only
    controls["T"]["timeslice"] = args.calibration_T_timeslice
    
    controls["G"]["first_selfcal"] = args.calibration_G_first_selfcal
    controls["G"]["timeslice"] = args.calibration_G_timeslice
    
    controls["B"]["first_selfcal"] = args.calibration_B_first_selfcal
    if args.calibration_B_timeslice is None:
        controls["B"]["timeslice"] = 1e5
    else:
        controls["B"]["timeslice"] = args.calibration_B_timeslice
    
    # Next we define a graph to run the continuum imaging pipeline
    result = ical_skymodel_list_rsexecute_workflow(
        bvis_list,  # List of BlockVisibilitys
        model_list,  # List of model images
        context=args.imaging_context,  # Use nifty-gridder
        threads=args.imaging_ng_threads,
        wstacking=args.imaging_w_stacking == "True",  # Correct for w term in gridding
        niter=args.clean_niter,  # iterations in minor cycle
        nmajor=args.clean_nmajor,  # Number of major cycles
        algorithm=args.clean_algorithm,
        gain=args.clean_gain,  # CLEAN loop gain
        scales=args.clean_scales,  # Scales for multi-scale cleaning
        fractional_threshold=args.clean_fractional_threshold,
        # Threshold per major cycle
        threshold=args.clean_threshold,  # Final stopping threshold
        nmoment=args.clean_nmoment,
        # Number of frequency moments (1 = no dependence)
        psf_support=args.clean_psf_support,
        # Support of PSF used in minor cycles (halfwidth in pixels)
        restored_output=args.clean_restored_output,  # Type of restored image
        deconvolve_facets=args.clean_facets,
        deconvolve_overlap=args.clean_overlap,
        deconvolve_taper=args.clean_taper,
        calibration_context=args.calibration_context,
        controls=controls,
        global_solution=args.calibration_global_solution,
        component_threshold=args.clean_component_threshold,
        component_method=args.clean_component_method,
        dft_compute_kernel=args.imaging_dft_kernel,
    )
    # Execute the Dask graph
    log.info("Starting compute of ICAL pipeline graph ")
    deconvolved, residual, restored, skymodel, gt_list = rsexecute.compute(
        result, sync=True
    )
    log.info("Finished compute of ICAL pipeline graph")
    
    imagename = msname.replace(".ms", "_nmoment{}_ical".format(args.clean_nmoment))
    return write_results(imagename, (deconvolved, residual, restored, skymodel))


def invert(args, bvis_list, model_list, msname):
    """Run invert

    :param args: The parameters read from the CLI using argparse
    :param bvis_list: A list of or graph to make BlockVisibilitys
    :param model_list: A list of or graph to make model images
    :param msname: The filename of the MeasurementSet
    :return: Names of output image (dirty image or psf image)
    """
    # Next we define a graph to run the continuum imaging pipeline
    result = invert_list_rsexecute_workflow(
        bvis_list,  # List of BlockVisibilitys
        model_list,  # List of model images
        context=args.imaging_context,
        dopsf=args.imaging_dopsf == "True",
        threads=args.imaging_ng_threads,
        wstacking=args.imaging_w_stacking == "True",
        dft_compute_kernel=args.imaging_dft_kernel,
    )
    result = sum_invert_results_rsexecute(result)
    # Execute the Dask graph
    log.info("Starting compute of invert graph ")
    dirty, sumwt = rsexecute.compute(result, sync=True)
    log.info("Finished compute of invert graph")
    imagename = msname.replace(".ms", "_invert")
    
    if args.imaging_dopsf == "True":
        log.info(qa_image(dirty, context="PSF"))
        show_image(dirty, title=f"{imagename} PSF image")
        plt.savefig(imagename + "_psf.png")
        plt.show(block=False)
        psfname = imagename + "_psf.fits"
        export_image_to_fits(dirty, psfname)
        return psfname
    else:
        log.info(qa_image(dirty, context="Dirty"))
        show_image(dirty, title=f"{imagename} Dirty image")
        plt.savefig(imagename + "_dirty.png")
        plt.show(block=False)
        dirtyname = imagename + "_dirty.fits"
        export_image_to_fits(dirty, dirtyname)
        return dirtyname


def main():
    # Get command line inputs
    parser = cli_parser()
    args = parser.parse_args()
    image = imager(args)


if __name__ == "__main__":
    main()
