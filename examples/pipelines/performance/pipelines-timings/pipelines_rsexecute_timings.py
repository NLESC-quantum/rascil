# Pipeline timings, using Dask
#
# This takes command line arguments and runs simulation and data reduction, outputing a CSV file of various
# timings.
#
import logging
import os
import pprint
import socket
import time
import csv
import seqfile

import argparse

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models import PolarisationFrame
from rascil.processing_components import (
    image_gather_channels,
    export_image_to_fits,
    qa_image,
    advise_wide_field,
    create_low_test_skymodel_from_gleam,
)
from rascil.processing_components import create_image
from rascil.processing_components.calibration.chain_calibration import (
    create_calibration_controls,
)
from rascil.processing_components.util.sizeof import get_size
from rascil.workflows import (
    invert_list_rsexecute_workflow,
    weight_list_rsexecute_workflow,
    taper_list_rsexecute_workflow,
    remove_sumwt,
    ical_list_rsexecute_workflow,
    simulate_list_rsexecute_workflow,
    corrupt_list_rsexecute_workflow,
    predict_skymodel_list_rsexecute_workflow,
)
from rascil.workflows.rsexecute.execution_support.rsexecute import (
    rsexecute,
    get_dask_client,
)


def init_logging():
    """ Initialise the logging.
    
    We need to run this function on all Dask workers
    """
    logging.basicConfig(
        filename="pipelines_rsexecute_timings.log",
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )


init_logging()
log = logging.getLogger("rascil-logger")

pp = pprint.PrettyPrinter()


def lprint(*args):
    s = pprint.pformat(*args)
    log.info(s)


def sort_dict(dc):
    newdc = dict()
    for d in sorted(dc):
        newdc[d] = dc[d]
    return dc


def git_hash():
    """Get the hash for this git repository.

    Requires that the code tree was created using git

    :return: string or "unknown"
    """
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"])
    except Exception as excp:
        log.info(excp)
        return "unknown"


def trial_case(
    results,
    seed=180555,
    context="ng",
    nworkers=4,
    threads_per_worker=4,
    memory=0,
    processes=True,
    order="frequency",
    nfreqwin=7,
    ntimes=3,
    rmax=750.0,
    use_dask=True,
    flux_limit=0.3,
    nmajor=5,
    dft_threshold=1.0,
    deconvolve_facets=8,
    deconvolve_overlap=16,
    deconvolve_taper="tukey",
    write_fits=False,
):
    """Single trial for performance-timings

    Simulates visibilities from GLEAM including phase errors
    Makes dirty image and PSF
    Runs ICAL pipeline

    The results are in a dictionary:

    'time overall',  overall execution time (s)
    'time predict', time to execute GLEAM prediction graph
    'time invert', time to make dirty image
    'time invert graph', time to make dirty image graph
    'time ICAL graph', time to create ICAL graph
    'time ICAL', time to execute ICAL graph
    'context', type of imaging e.g. 'ng'
    'nworkers', number of workers to create
    'threads_per_worker',
    'nnodes', Number of nodes,
    'processes', 'order', Ordering of data_models
    'nfreqwin', Number of frequency windows in simulation
    'ntimes', Number of hour angles in simulation
    'rmax', Maximum radius of stations used in simulation (m)
    'npixel', Number of pixels in image
    'cellsize', Cellsize in radians
    'seed', Random number seed
    'dirty_max', Maximum in dirty image
    'dirty_min', Minimum in dirty image
    'restored_max',
    'restored_min',
    'deconvolved_max',
    'deconvolved_min',
    'residual_max',
    'residual_min',
    'git_info', GIT hash (not definitive since local mods are possible)

    :param results: Initial state
    :param seed: Random number seed (used in gain simulations)
    :param context: imaging context
    :param context: Type of context: '2d'|'ng'
    :param nworkers: Number of dask workers to use
    :param threads_per_worker: Number of threads per worker
    :param processes: Use processes instead of threads 'processes'|'threads'
    :param order: See simulate_list_list_rsexecute_workflow_workflowkflow
    :param nfreqwin: See simulate_list_list_rsexecute_workflow_workflowkflow
    :param ntimes: See simulate_list_list_rsexecute_workflow_workflowkflow
    :param rmax: See simulate_list_list_rsexecute_workflow_workflowkflow
    :param use_dask: Use dask or immediate evaluation
    :return: results dictionary
    """
    if use_dask:
        scheduler = os.getenv("RASCIL_DASK_SCHEDULER", None)
        if scheduler is not None:
            client = get_dask_client(
                n_workers=nworkers,
                memory_limit=memory * 1024 * 1024 * 1024,
                threads_per_worker=threads_per_worker,
            )
            rsexecute.set_client(client=client)
        else:
            rsexecute.set_client(use_dask=True)
    else:
        rsexecute.set_client(use_dask=use_dask)
        results["nnodes"] = 1

    rsexecute.run(init_logging)

    lprint("Starting pipelines_rsexecute_timings")

    numpy.random.seed(seed)
    results["seed"] = seed

    start_all = time.time()

    results["context"] = context
    results["hostname"] = socket.gethostname()
    results["git_hash"] = git_hash()
    results["epoch"] = time.strftime("%Y-%m-%d %H:%M:%S")

    lprint("Context is %s" % context)

    results["nworkers"] = nworkers
    results["threads_per_worker"] = threads_per_worker
    results["processes"] = processes
    results["memory"] = memory
    results["order"] = order
    results["nfreqwin"] = nfreqwin
    results["ntimes"] = ntimes
    results["rmax"] = rmax
    results["dft threshold"] = dft_threshold

    results["use_dask"] = use_dask

    lprint("At start, configuration is:")
    lprint(sort_dict(results))

    # Parameters determining scale of simulation.
    frequency = numpy.linspace(1.0e8, 1.2e8, nfreqwin)
    centre = nfreqwin // 2
    if nfreqwin > 1:
        channel_bandwidth = numpy.array(nfreqwin * [frequency[1] - frequency[0]])
    else:
        channel_bandwidth = numpy.array([1e6])

    times = numpy.linspace(-numpy.pi / 4.0, numpy.pi / 4.0, ntimes)
    phasecentre = SkyCoord(
        ra=+0.0 * u.deg, dec=-40.0 * u.deg, frame="icrs", equinox="J2000"
    )

    lprint("****** Visibility creation ******")
    # Create the empty BlockVisibility's and persist these on the cluster
    bvis_list = simulate_list_rsexecute_workflow(
        "LOWBD2",
        frequency=frequency,
        channel_bandwidth=channel_bandwidth,
        times=times,
        phasecentre=phasecentre,
        order=order,
        format="blockvis",
        rmax=rmax,
    )
    bvis_list = rsexecute.persist(bvis_list)

    # Find the best imaging parameters but don't bring the vis_list back here
    lprint("****** Finding wide field parameters ******")
    advice = [
        rsexecute.execute(advise_wide_field)(
            v,
            guard_band_image=6.0,
            delA=0.02,
            oversampling_synthesised_beam=4.0,
            verbose=False,
        )
        for v in bvis_list
    ]

    advice = rsexecute.compute(advice, sync=True)[-1]

    # Deconvolution via sub-images requires 2^n
    npixel = advice["npixels2"]
    results["npixel"] = npixel
    cellsize = advice["cellsize"]
    results["cellsize"] = cellsize
    lprint(
        "Image will have %d by %d pixels, cellsize = %.6f rad"
        % (npixel, npixel, cellsize)
    )

    # Create an empty model image
    model_list = [
        rsexecute.execute(create_image)(
            npixel=npixel,
            cellsize=cellsize,
            frequency=[frequency[f]],
            channel_bandwidth=[channel_bandwidth[f]],
            phasecentre=phasecentre,
            polarisation_frame=PolarisationFrame("stokesI"),
        )
        for f, freq in enumerate(frequency)
    ]
    model_list = rsexecute.persist(model_list)

    lprint("****** Setting up imaging parameters ******")
    # Now set up the imaging parameters

    if context == "2d":
        lprint("Using RASCIL 2D gridder")
    elif context == "ng":
        lprint("Using Nifty Gridder")
    else:
        log.error(f"context {context} no longer supported")

    # Make a skymodel from gleam, with bright sources as components and weak sources in an image
    lprint("****** Starting GLEAM skymodel creation ******")
    skymodel_list = [
        rsexecute.execute(create_low_test_skymodel_from_gleam)(
            npixel=npixel,
            cellsize=cellsize,
            frequency=[frequency[f]],
            phasecentre=phasecentre,
            polarisation_frame=PolarisationFrame("stokesI"),
            flux_limit=flux_limit,
            flux_threshold=dft_threshold,
            flux_max=5.0,
        )
        for f, freq in enumerate(frequency)
    ]

    skymodel_list = rsexecute.persist(skymodel_list)

    # We use predict_skymodel so that we can use skycomponents as well as images
    lprint("****** Starting GLEAM skymodel prediction ******")
    predicted_bvis_list = [
        predict_skymodel_list_rsexecute_workflow(
            bvis_list[f],
            [skymodel_list[f]],
            context=context,
        )[0]
        for f, freq in enumerate(frequency)
    ]

    # Corrupt the visibility for the GLEAM model
    lprint("****** Visibility corruption ******")
    corrupted_bvis_list = corrupt_list_rsexecute_workflow(
        predicted_bvis_list, phase_error=1.0, seed=seed
    )
    lprint("****** Weighting and tapering ******")
    corrupted_bvis_list = weight_list_rsexecute_workflow(
        corrupted_bvis_list, model_list
    )
    corrupted_bvis_list = taper_list_rsexecute_workflow(
        corrupted_bvis_list, 0.003 * 750.0 / rmax
    )
    corrupted_bvis_list = rsexecute.persist(corrupted_bvis_list)

    lprint("****** Starting dirty image calculation ******")
    start = time.time()
    dirty_list = invert_list_rsexecute_workflow(
        corrupted_bvis_list, model_list, context=context
    )
    results["size invert graph"] = get_size(dirty_list)
    lprint("Size of dirty graph is %.3E bytes" % (results["size invert graph"]))
    end = time.time()
    results["time invert graph"] = end - start
    lprint("Construction of invert graph took %.3f seconds" % (end - start))

    start = time.time()
    dirty, sumwt = rsexecute.compute(dirty_list, sync=True)[centre]
    end = time.time()
    results["time invert"] = end - start
    lprint("Dirty image invert took %.3f seconds" % (end - start))
    lprint(
        "Maximum in dirty image is %f, sumwt is %s"
        % (numpy.max(numpy.abs(dirty["pixels"].data)), str(sumwt))
    )
    qa = qa_image(dirty)
    results["dirty_max"] = qa.data["max"]
    results["dirty_min"] = qa.data["min"]
    if write_fits:
        export_image_to_fits(
            dirty, "pipelines_rsexecute_timings-%s-dirty.fits" % context
        )

    # Create the ICAL pipeline to run major cycles, starting selfcal at cycle 1. A global solution across all
    # frequencies (i.e. Visibilities) is performed.

    if deconvolve_facets > 1:
        lprint("Using subimage clean")

    lprint("****** Starting ICAL graph creation ******")

    controls = create_calibration_controls()

    controls["T"]["first_selfcal"] = 1
    controls["T"]["timeslice"] = "auto"

    start = time.time()
    ical_list = ical_list_rsexecute_workflow(
        corrupted_bvis_list,
        model_imagelist=model_list,
        context=context,
        scales=[0, 3, 10],
        algorithm="mmclean",
        nmoment=3,
        niter=1000,
        fractional_threshold=0.1,
        threshold=0.01,
        nmajor=nmajor,
        gain=0.25,
        psf_support=128,
        deconvolve_facets=deconvolve_facets,
        deconvolve_overlap=deconvolve_overlap,
        deconvolve_taper=deconvolve_taper,
        timeslice="auto",
        global_solution=True,
        do_selfcal=True,
        calibration_context="T",
        controls=controls,
    )

    results["size ICAL graph"] = get_size(ical_list)
    lprint("Size of ICAL graph is %.3E bytes" % results["size ICAL graph"])
    end = time.time()
    results["time ICAL graph"] = end - start
    lprint("Construction of ICAL graph took %.3f seconds" % (end - start))

    # Execute the graph
    lprint("****** Executing ICAL graph ******")
    rsexecute.init_statistics()

    start = time.time()
    deconvolved, residual, restored, gaintables = rsexecute.compute(
        ical_list, sync=True
    )
    end = time.time()

    perf = rsexecute.save_statistics("pipelines_rsexecute_timings_%s_ical" % context)
    results["total"] = perf["total"]
    results["duration"] = perf["duration"]
    results["speedup"] = perf["speedup"]

    results["time ICAL"] = end - start
    lprint("ICAL graph execution took %.3f seconds" % (end - start))
    qa = qa_image(deconvolved[centre])
    results["deconvolved_max"] = qa.data["max"]
    results["deconvolved_min"] = qa.data["min"]
    if write_fits:
        deconvolved_cube = image_gather_channels(deconvolved)
        export_image_to_fits(
            deconvolved_cube,
            "pipelines_rsexecute_timings-%s-ical_deconvolved.fits" % context,
        )

    qa = qa_image(residual[centre][0])
    results["residual_max"] = qa.data["max"]
    results["residual_min"] = qa.data["min"]
    if write_fits:
        residual_cube = remove_sumwt(residual)
        residual_cube = image_gather_channels(residual_cube)
        export_image_to_fits(
            residual_cube, "pipelines_rsexecute_timings-%s-ical_residual.fits" % context
        )

    qa = qa_image(restored[centre])
    results["restored_max"] = qa.data["max"]
    results["restored_min"] = qa.data["min"]
    if write_fits:
        restored_cube = image_gather_channels(restored)
        export_image_to_fits(
            restored_cube, "pipelines_rsexecute_timings-%s-ical_restored.fits" % context
        )
    #
    rsexecute.close()

    end_all = time.time()
    results["time overall"] = end_all - start_all

    lprint("At end, results are:")
    results = sort_dict(results)
    lprint(results)

    return results


def write_results(filename, fieldnames, results):
    """Write results to a csv file.

    :param filename: Name of existing csvfile
    :param fieldnames: field names
    :param results: dictionary of results
    :return:
    """
    with open(filename, "a") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
            delimiter=",",
            quotechar="|",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writerow(results)
        csvfile.close()


def write_header(filename, fieldnames):
    """Write a header for a csvfile

    :param filename: csv file to be written
    :param fieldnames:
    :return:
    """
    with open(filename, "w") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
            delimiter=",",
            quotechar="|",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        csvfile.close()


def process(args):
    results = {}

    results["jobid"] = args.jobid

    nworkers = args.nworkers
    results["nworkers"] = nworkers

    context = args.context
    results["context"] = context

    nnodes = args.nnodes
    results["nnodes"] = nnodes

    threads_per_worker = args.nthreads

    lprint("Using %s workers" % nworkers)
    lprint("Using %s threads per worker" % threads_per_worker)

    nfreqwin = args.nfreqwin
    results["nfreqwin"] = nfreqwin

    rmax = args.rmax
    results["rmax"] = rmax

    flux_limit = args.flux_limit
    results["flux_limit"] = flux_limit

    dft_threshold = args.dft_threshold
    results["dft threshold"] = dft_threshold

    context = args.context
    results["context"] = context

    memory = args.memory
    results["memory"] = memory

    ntimes = args.ntimes
    results["ntimes"] = ntimes

    nmajor = args.nmajor
    results["nmajor"] = nmajor

    deconvolve_facets = args.deconvolve_facets
    deconvolve_overlap = args.deconvolve_overlap
    deconvolve_taper = args.deconvolve_taper

    results["hostname"] = socket.gethostname()
    results["epoch"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results["driver"] = "pipelines_rsexecute_timings"

    use_dask = args.use_dask == "True"
    if use_dask:
        lprint("Using Dask")

    threads_per_worker = args.nthreads

    write_fits = args.write_fits == "True"

    lprint("Defining %d frequency windows" % nfreqwin)

    fieldnames = [
        "cellsize",
        "context",
        "deconvolved_max",
        "deconvolved_min",
        "deconvolve_facets",
        "deconvolve_overlap",
        "deconvolve_taper",
        "dft threshold",
        "dirty_max",
        "dirty_min",
        "driver",
        "duration",
        "epoch",
        "flux_limit",
        "git_hash",
        "hostname",
        "jobid",
        "memory",
        "nfreqwin",
        "nmajor",
        "nnodes",
        "npixel",
        "ntimes",
        "nworkers",
        "order",
        "processes",
        "residual_max",
        "residual_min",
        "restored_max",
        "restored_min",
        "rmax",
        "seed",
        "size ICAL graph",
        "size invert graph",
        "speedup",
        "threads_per_worker",
        "time ICAL",
        "time ICAL graph",
        "time invert",
        "time invert graph",
        "time overall",
        "total",
        "use_dask",
    ]

    filename = seqfile.findNextFile(
        prefix="%s_%s_" % (results["driver"], results["hostname"]), suffix=".csv"
    )
    log.info("Saving results to %s" % filename)

    write_header(filename, fieldnames)

    results = trial_case(
        results,
        use_dask=use_dask,
        nworkers=nworkers,
        rmax=rmax,
        context=context,
        memory=memory,
        threads_per_worker=threads_per_worker,
        nfreqwin=nfreqwin,
        ntimes=ntimes,
        flux_limit=flux_limit,
        nmajor=nmajor,
        dft_threshold=dft_threshold,
        deconvolve_facets=deconvolve_facets,
        deconvolve_overlap=deconvolve_overlap,
        deconvolve_taper=deconvolve_taper,
        write_fits=write_fits,
    )
    write_results(filename, fieldnames, results)

    lprint("Exiting %s" % results["driver"])


def main():

    parser = argparse.ArgumentParser(
        description="Benchmark pipelines in numpy and dask"
    )
    parser.add_argument("--use_dask", type=str, default="True", help="Use Dask?")
    parser.add_argument("--nnodes", type=int, default=None, help="Number of nodes")
    parser.add_argument("--nthreads", type=int, default=4, help="Number of threads")
    parser.add_argument("--memory", type=int, default=0, help="Memory per worker")
    parser.add_argument("--nworkers", type=int, default=4, help="Number of workers")

    parser.add_argument("--nmajor", type=int, default=5, help="Number of major cycles")

    parser.add_argument("--ntimes", type=int, default=7, help="Number of hour angles")
    parser.add_argument(
        "--nfreqwin", type=int, default=16, help="Number of frequency windows"
    )
    parser.add_argument(
        "--context",
        type=str,
        default="ng",
        help="Imaging context: 2d|ng",
    )
    parser.add_argument(
        "--rmax", type=float, default=750.0, help="Maximum baseline (m)"
    )
    parser.add_argument("--jobid", type=int, default=0, help="JOBID from slurm")
    parser.add_argument(
        "--flux_limit", type=float, default=0.3, help="Flux limit for components"
    )

    parser.add_argument(
        "--deconvolve_facets",
        type=int,
        default=4,
        help="Number of facets for deconvolution",
    )
    parser.add_argument(
        "--deconvolve_overlap",
        type=int,
        default=16,
        help="Number of pixels overlap for deconvolution",
    )
    parser.add_argument(
        "--deconvolve_taper",
        type=str,
        default="tukey",
        help="Facet taper for deconvolution",
    )

    parser.add_argument(
        "--dft_threshold", type=float, default=1.0, help="Flux above which DFT is used"
    )
    parser.add_argument(
        "--write_fits", type=str, default="True", help="Write FITS files??"
    )

    process(parser.parse_args())

    exit()


if __name__ == "__main__":
    main()
