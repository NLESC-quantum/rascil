"""Simulation of the effect of errors on MID observations

This measures the change in a dirty imagethe induced by various errors:
    - The sky can be a point source at the half power point or a realistic sky constructed from S3-SEX catalog.
    - The observation is by MID over a range of hour angles
    - Processing can be divided into chunks of time (default 1800s)
    - Dask is used to distribute the processing over a number of workers.
    - Various plots are produced, The primary output is a csv file containing information about the statistics of
    the residual images.

"""
import logging
import os
import pprint
import sys

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from distributed import Client

from rascil.data_models import SkyModel, rascil_data_path, PolarisationFrame, export_skycomponent_to_hdf5
from rascil.processing_components import export_image_to_fits, qa_image, create_named_configuration, \
    create_mid_simulation_components
from rascil.processing_components.imaging.base import create_image_from_visibility, advise_wide_field
from rascil.processing_components.imaging.primary_beams import create_vp
from rascil.processing_components.simulation.simulation_helpers import find_pb_width_null
from rascil.processing_components.visibility import qa_visibility, create_blockvisibility
from rascil.processing_components.visibility.base import export_blockvisibility_to_ms, copy_visibility
from rascil.workflows.rsexecute import invert_list_rsexecute_workflow, sum_invert_results_rsexecute, \
    weight_list_rsexecute_workflow
from rascil.workflows.rsexecute.execution_support import rsexecute, get_dask_client
from rascil.workflows.rsexecute.imaging.imaging_rsexecute import sum_predict_results_rsexecute, \
    subtract_list_rsexecute_workflow
from rascil.workflows.rsexecute.simulation.simulation_rsexecute import \
    create_surface_errors_gaintable_rsexecute_workflow, \
    create_pointing_errors_gaintable_rsexecute_workflow, \
    create_polarisation_gaintable_rsexecute_workflow, \
    create_atmospheric_errors_gaintable_rsexecute_workflow, \
    create_heterogeneous_gaintable_rsexecute_workflow
from rascil.workflows.rsexecute.skymodel.skymodel_rsexecute import predict_skymodel_list_rsexecute_workflow

log = logging.getLogger("rascil-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


def make_images_workflow(args, bvis_list, state):
    if args.imaging_context not in ['2d', 'ng']:
        return bvis_list
    else:
        imaging_context = args.imaging_context
    
    cellsize = args.cellsize
    npixel = args.npixel
    advice = [rsexecute.execute(advise_wide_field)(bvis, verbose=False) for bvis in bvis_list]
    advice = rsexecute.compute(advice, sync=True)
    if cellsize is None:
        cellsize = advice[0]["cellsize"]
    if npixel is None:
        npixel = advice[0]["npixels"]
    model_list = [rsexecute.execute(create_image_from_visibility)
                  (bvis, npixel=npixel, cellsize=cellsize,
                   polarisation_frame=PolarisationFrame(args.image_pol))
                  for bvis in bvis_list]
    model_list = rsexecute.persist(model_list)
    bvis_list = weight_list_rsexecute_workflow(bvis_list, model_list, weighting=args.weighting,
                                               robustness=args.robustness)
    dirty_list = invert_list_rsexecute_workflow(bvis_list, template_model_imagelist=model_list, context=imaging_context)
    dirty_list = sum_invert_results_rsexecute(dirty_list)
    result = rsexecute.compute(dirty_list, sync=True)
    dirty, sumwt = result
    log.info(qa_image(dirty, context=state))
    dirtyname = "{results}/SKA_{configuration}_SIM_{duration}_{band}_dec_{dec:.1f}_{mode}_{state}_dirty.fits". \
        format(configuration=args.configuration, results=args.results, band=args.band, dec=args.declination,
               duration=args.duration, mode=args.mode,
               state=state)
    log.info("Writing dirty image {}".format(dirtyname))
    export_image_to_fits(dirty, dirtyname)
    return bvis_list


def simulation(args):
    pp = pprint.PrettyPrinter()
    pp.pprint(vars(args))
    
    ra = args.ra
    declination = args.declination
    mode = args.mode
    band = args.band
    
    if args.imaging_context not in ['2d', 'ng']:
        args.imaging_context = None
    else:
        args.imaging_context = args.imaging_context
    
    if args.duration == "short":
        integration_time = 1.0
        time_range = [-180.0 / 3600.0, 180.0 / 3600.0]
        time_chunk = 12.0
    elif args.duration == "medium":
        integration_time = 10.0
        time_range = [-0.5, 0.5]
        time_chunk = 100.0
    elif args.duration == "long":
        integration_time = 60.0
        time_range = [-4.0, 4.0]
        if mode == "surface":
            time_chunk = integration_time
        else:
            time_chunk = 1800.0
    else:
        args.duration = "custom"
        integration_time = args.integration_time
        time_range = args.time_range
        time_chunk = args.time_chunk
    
    logfile = "{results}/SKA_{configuration}_SIM_{duration}_{band}_dec_{dec:.1f}_{mode}.log". \
        format(configuration=args.configuration, results=args.results, band=band, dec=declination,
               duration=args.duration, mode=mode)
    print("Writing log file to {}".format(logfile))
    
    def init_logging():
        logging.basicConfig(filename=logfile,
                            filemode='a',
                            format='%(asctime)s.%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)
    
    init_logging()
    log = logging.getLogger('rascil-logger')
    
    log.info("Starting simulation of {}".format(args.mode))
    
    initialise_rsexecute(args)
    
    rsexecute.run(init_logging)
    rsexecute.init_statistics()
    
    image_polarisation_frame = PolarisationFrame(args.image_pol)
    vis_polarisation_frame = PolarisationFrame(args.vis_pol)
    log.info("Image polarisation: {}".format(str(image_polarisation_frame)))
    log.info("Vis polarisation: {}".format(str(vis_polarisation_frame)))
    use_radec = args.use_radec == "True"
    
    log.info("Simulating {duration} observation: {time_range} hours in integrations of {integration_time}s"
             .format(duration=args.duration, time_range=time_range, integration_time=integration_time))
    log.info("Simulations processed in chunks of {:.1f} seconds".format(time_chunk))
    
    pbtype = args.pbtype
    rmax = args.rmax
    flux_limit = args.flux_limit
    vp_directory = args.vp_directory
    
    seed = args.seed
    basename = os.path.basename(os.getcwd())
    
    # Set up details of simulated observation
    nchan = args.nchan
    channel_width = args.channel_width
    if band == 'B1':
        if channel_width is None:
            channel_width = 1.0e8 / nchan
        frequency = 0.7650e9 + numpy.arange(nchan) * channel_width
    elif band == 'B1LOW':
        if channel_width is None:
            channel_width = 1.0e8 / nchan
        frequency = 0.350e9 + numpy.arange(nchan) * channel_width
    elif band == 'B2':
        if channel_width is None:
            channel_width = 1.0e8 / nchan
        frequency = 1.36e9 + numpy.arange(nchan) * channel_width
    elif band == 'Ku':
        if channel_width is None:
            channel_width = 1.0e9 / nchan
        frequency = 12.179e9 + numpy.arange(nchan) * channel_width
    else:
        raise ValueError("Unknown band %s" % band)
    
    frequency = numpy.array(frequency)
    channel_bandwidth = numpy.repeat(channel_width, len(frequency))
    
    phasecentre = SkyCoord(ra=ra * u.deg, dec=declination * u.deg, frame='icrs', equinox='J2000')
    
    mid = create_named_configuration(args.configuration, rmax=rmax)
    
    times = numpy.arange(time_range[0] * 3600, time_range[1] * 3600, integration_time)
    times *= numpy.pi / 43200.0
    
    entire_bvis = create_blockvisibility(mid, times, frequency=frequency,
                                         channel_bandwidth=channel_bandwidth,
                                         weight=1.0,
                                         phasecentre=phasecentre,
                                         polarisation_frame=vis_polarisation_frame)
    time_slices = len(numpy.arange(time_range[0] * 3600, time_range[1] * 3600, time_chunk))
    bvis_list = visibility_scatter_time(entire_bvis, time_slices)
    assert len(bvis_list) == time_slices
    entire_bvis = rsexecute.scatter(entire_bvis)
    bvis_list = rsexecute.scatter(bvis_list)
    
    # We need the HWHM of the primary beam, and the location of the nulls
    hwhm_deg, null_az_deg, null_el_deg = find_pb_width_null(pbtype, frequency)
    
    hwhm = hwhm_deg * numpy.pi / 180.0
    
    fov_deg = 8.0 * 1.36e9 / frequency[0]
    
    pb_npixel = 256
    d2r = numpy.pi / 180.0
    pb_cellsize = d2r * fov_deg / pb_npixel
    pbradius = args.pbradius * hwhm
    
    # We need different components for each frequency but not for each time. We compute this immediately
    # since we need to know the sizes of the component lists below. Note that the component lists may be
    # of different length
    original_components = create_mid_simulation_components(phasecentre, frequency, flux_limit,
                                                           pbradius, pb_npixel, pb_cellsize,
                                                           show=False, fov=10,
                                                           polarisation_frame=image_polarisation_frame,
                                                           flux_max=10.0)
    
    comps_name = "{results}/SKA_{configuration}_SIM_{band}_dec_{dec:.1f}_components.hdf5". \
        format(configuration=args.configuration, results=args.results, band=band, dec=declination)
    export_skycomponent_to_hdf5(original_components[0], comps_name)
    
    print(original_components[0])
    
    # Now create the voltage patterns one per blockvis
    vp_list = [rsexecute.execute(create_image_from_visibility)(bv, npixel=pb_npixel,
                                                               nchan=1, cellsize=pb_cellsize,
                                                               phasecentre=phasecentre,
                                                               polarisation_frame=image_polarisation_frame,
                                                               override_cellsize=False) for bv in bvis_list]
    vp_list = [rsexecute.execute(create_vp)(vp, pbtype, pointingcentre=phasecentre, use_local=not use_radec)
               for vp in vp_list]
    future_vp_list = rsexecute.persist(vp_list)
    
    # End of setup, now we generate the gaintables with and without errors
    
    if mode == 'random_pointing':
        
        # Random pointing errors
        global_pointing_error = numpy.array(args.global_pe)
        static_pointing_error = numpy.array(args.static_pe)
        pointing_error = args.dynamic_pe
        
        a2r = numpy.pi / (3600.0 * 1800)
        
        no_error_gtl, error_gtl = \
            create_pointing_errors_gaintable_rsexecute_workflow(bvis_list,
                                                                original_components,
                                                                sub_vp_list=future_vp_list,
                                                                use_radec=use_radec,
                                                                pointing_error=a2r * pointing_error,
                                                                static_pointing_error=a2r * static_pointing_error,
                                                                global_pointing_error=a2r * global_pointing_error,
                                                                seed=seed,
                                                                show=False, basename=mode)
    elif mode == 'wind_pointing':
        # Wind-induced pointing errors
        no_error_gtl, error_gtl = \
            create_pointing_errors_gaintable_rsexecute_workflow(bvis_list,
                                                                original_components,
                                                                sub_vp_list=future_vp_list,
                                                                use_radec=use_radec,
                                                                time_series='wind',
                                                                time_series_type=args.wind_conditions,
                                                                seed=seed,
                                                                show=False, basename=mode)
    elif mode == 'troposphere':
        no_error_gtl, error_gtl = \
            create_atmospheric_errors_gaintable_rsexecute_workflow(bvis_list,
                                                                   original_components,
                                                                   r0=args.r0,
                                                                   screen=args.screen,
                                                                   height=args.height,
                                                                   type_atmosphere=args.mode,
                                                                   show=args.show == "True",
                                                                   basename=mode,
                                                                   reference=True)
    elif mode == 'ionosphere':
        no_error_gtl, error_gtl = \
            create_atmospheric_errors_gaintable_rsexecute_workflow(bvis_list,
                                                                   original_components,
                                                                   r0=args.r0,
                                                                   screen=args.screen,
                                                                   height=args.height,
                                                                   type_atmosphere=args.mode,
                                                                   show=args.show == "True",
                                                                   basename=mode,
                                                                   reference=True)
    
    elif mode == 'surface':
        # Dish surface sag due to gravity
        no_error_gtl, error_gtl = \
            create_surface_errors_gaintable_rsexecute_workflow(band, bvis_list,
                                                               original_components,
                                                               vp_directory=vp_directory, use_radec=use_radec,
                                                               show=False, basename=basename,
                                                               elevation_sampling=args.elevation_sampling)
    elif mode == 'heterogeneous':
        # Different antennas
        no_error_gtl, error_gtl = \
            create_heterogeneous_gaintable_rsexecute_workflow(band, bvis_list,
                                                              original_components,
                                                              use_radec=use_radec,
                                                              show=False, basename=basename)
    elif mode == 'polarisation':
        # Polarised beams
        no_error_gtl, error_gtl = \
            create_polarisation_gaintable_rsexecute_workflow(band, bvis_list,
                                                             original_components,
                                                             basename=basename,
                                                             show=False)
    else:
        raise ValueError("Unknown type of error %s" % mode)
    
    # Now generate skymodels using the same components but different gaintable. Note that there is a
    # different skymodel for each blockvis
    error_sm_list = [[
        rsexecute.execute(SkyModel, nout=1)(components=[original_components[i]],
                                            gaintable=error_gtl[ibv][i])
        for i, _ in enumerate(original_components)] for ibv, bv in enumerate(bvis_list)]
    
    no_error_sm_list = [[
        rsexecute.execute(SkyModel, nout=1)(components=[original_components[i]],
                                            gaintable=no_error_gtl[ibv][i])
        for i, _ in enumerate(original_components)] for ibv, bv in enumerate(bvis_list)]
    
    # Predict_skymodel_list_rsexecute_workflow calculates the BlockVis for each of a list of
    # SkyModels. We want to add these across SkyModels and then concatenate BlockVis
    error_bvis_list = [rsexecute.execute(copy_visibility)(bvis, zero=True) for bvis in bvis_list]
    error_bvis_list = \
        [sum_predict_results_rsexecute(predict_skymodel_list_rsexecute_workflow(bvis, error_sm_list[ibvis],
                                                                                docal=True, context='2d'))
         for ibvis, bvis in enumerate(error_bvis_list)]
    
    no_error_bvis_list = [rsexecute.execute(copy_visibility)(bvis, zero=True) for bvis in bvis_list]
    no_error_bvis_list = \
        [sum_predict_results_rsexecute(predict_skymodel_list_rsexecute_workflow(bvis, no_error_sm_list[ibvis],
                                                                                docal=True, context='2d'))
         for ibvis, bvis in enumerate(no_error_bvis_list)]
    
    def concat_write(bvis, state):
        msname = "{results}/SKA_{configuration}_SIM_{duration}_{band}_dec_{dec:.1f}_{mode}_{state}.ms". \
            format(configuration=args.configuration, results=args.results, band=band, dec=declination,
                   duration=args.duration, mode=mode, state=state)
        log.info("Writing {mode} {state} visibility to {msname}".format(mode=mode, state=state, msname=msname))
        log.info(qa_visibility(bvis, context="{} visibility".format(state)))
        export_blockvisibility_to_ms(msname, [bvis])
        return bvis
    
    error_bvis_list = make_images_workflow(args, error_bvis_list, "on")
    error_bvis = rsexecute.execute(copy_visibility)(entire_bvis)
    error_bvis = rsexecute.execute(visibility_gather_time)(error_bvis_list, error_bvis, vis_slices=time_slices)
    error_bvis = rsexecute.execute(concat_write, nout=1)(error_bvis, "on")
    
    no_error_bvis_list = make_images_workflow(args, no_error_bvis_list, "off")
    no_error_bvis = rsexecute.execute(copy_visibility)(entire_bvis)
    no_error_bvis = rsexecute.execute(visibility_gather_time)(no_error_bvis_list, no_error_bvis, vis_slices=time_slices)
    no_error_bvis = rsexecute.execute(concat_write, nout=1)(no_error_bvis, "off")
    
    difference_bvis_list = subtract_list_rsexecute_workflow(error_bvis_list, no_error_bvis_list)
    difference_bvis_list = make_images_workflow(args, difference_bvis_list, "difference")
    difference_bvis_list = rsexecute.compute(difference_bvis_list, sync=True)
    
    difference_bvis = subtract_list_rsexecute_workflow([error_bvis], [no_error_bvis])[0]
    difference_bvis = rsexecute.execute(concat_write)(difference_bvis, "difference")
    difference_bvis = rsexecute.compute(difference_bvis, sync=True)
    print(difference_bvis)
    
    rsexecute.save_statistics("{results}/SKA_{configuration}_SIM_{duration}_{band}_dec_{dec:.1f}_{mode}". \
                              format(configuration=args.configuration, results=args.results, band=band,
                                     dec=declination, duration=args.duration, mode=mode))
    
    return True


def initialise_rsexecute(args):
    if args.use_dask == "True":
        if args.use_slurm == "True":
            try:
                import dask_jobqueue
                from dask_jobqueue import SLURMCluster
            except ImportError:
                raise Exception("dask_jobqueue.SLURMCluster is not available")
            
            cluster = SLURMCluster(n_workers=0,
                                   cores=args.cores,
                                   threads_per_worker=args.nthreads,
                                   processes=args.processes,
                                   memory=args.memory,
                                   project=args.slurm_project,
                                   walltime=args.slurm_walltime,
                                   queue=args.slurm_queue,
                                   local_directory=args.results,
                                   log_directory=args.results)
            print("Creating SLURMCluster and Dask Client")
            # cluster.adapt(minimum_jobs=1, maximum_jobs=args.nworkers)
            cluster.scale(args.nworkers)
            print("Scaling to {} workers".format(args.nworkers))
            print("For one SLURM job, the script is: {}".format(cluster.job_script()))
            client = Client(cluster)
            addr = client.scheduler_info()['address']
            services = client.scheduler_info()['services']
            if 'bokeh' in services.keys():
                bokeh_addr = 'http:%s:%s' % (addr.split(':')[1], services['bokeh'])
                print('Diagnostic pages available on port %s' % bokeh_addr)
            if 'dashboard' in services.keys():
                db_addr = 'http:%s:%s' % (addr.split(':')[1], services['dashboard'])
                print('Diagnostic pages available on port %s' % db_addr)
        
        else:
            client = get_dask_client(n_workers=args.nworkers,
                                     threads_per_worker=args.nthreads)
        rsexecute.set_client(use_dask=True, client=client)
    else:
        rsexecute.set_client(use_dask=False)


def cli_parser():
    global parser
    import argparse

    parser = argparse.ArgumentParser(description='Simulate SKA-MID direction dependent errors')
    parser.add_argument('--context', type=str, default='s3sky', help='s3sky or singlesource or null')
    # Observation definition
    parser.add_argument('--ra', type=float, default=0.0, help='Right ascension (degrees)')
    parser.add_argument('--declination', type=float, default=-40.0, help='Declination (degrees)')
    parser.add_argument('--rmax', type=float, default=1e4, help='Maximum distance of station from centre (m)')
    parser.add_argument('--band', type=str, default='B2', help="Band")
    parser.add_argument('--configuration', type=str, default='MID', help="Configuration: MID | MEERKAT+")
    parser.add_argument('--nchan', type=int, default=1, help="Number of frequency channels")
    parser.add_argument('--channel_width', type=float, default=None, help='Channel bandwidth (Hz)')
    parser.add_argument('--integration_time', type=float, default=180, help='Integration time (s)')
    parser.add_argument('--time_range', type=float, nargs=2, default=[-4.0, 4.0], help='Time range in hour angle')
    parser.add_argument('--image_pol', type=str, default='stokesIQUV', help='RASCIL polarisation frame for image')
    parser.add_argument('--vis_pol', type=str, default='linear',
                        help='RASCIL polarisation frame for visibility')
    parser.add_argument('--imaging_context', type=str, default=None, help="Imaging context '2d' | 'ng'")
    parser.add_argument('--npixel', type=int, default=None, help='Number of pixels')
    parser.add_argument('--cellsize', type=float, default=None, help='Cellsize in radians')
    parser.add_argument('--weighting', type=str, default='robust', help='Type of weighting')
    parser.add_argument('--robustness', type=float, default=0.0, help='Robustness for robust weighting')
    parser.add_argument('--pbradius', type=float, default=1.5, help='Radius of sources to include (in HWHM)')
    parser.add_argument('--pbtype', type=str, default='MID', help='Primary beam model: MID or MID_GAUSS')
    parser.add_argument('--seed', type=int, default=18051955, help='Random number seed')
    parser.add_argument('--flux_limit', type=float, default=0.01, help='Flux limit (Jy)')
    # Control parameters
    parser.add_argument('--use_radec', type=str, default="False", help='Calculate in RADEC (false)?')
    parser.add_argument('--shared_directory', type=str, default=rascil_data_path('configurations'),
                        help='Location of configuration files')
    parser.add_argument('--results', type=str, default='./', help='Directory for results')
    parser.add_argument('--elevation_sampling', type=float, default=1.0, help='Sampling in elevation for surface (deg)')
    # Noniso parameters
    parser.add_argument('--r0', type=float, default=5e3, help='R0 (meters)')
    parser.add_argument('--height', type=float, default=3e5, help='Height of layer (meters)')
    parser.add_argument('--screen', type=str, default=rascil_data_path('models/test_mpc_screen.fits'),
                        help='Location of atmospheric phase screen')
    # Dask parameters
    parser.add_argument('--nthreads', type=int, default=1, help='Number of threads')
    parser.add_argument('--processes', type=int, default=1, help='Number of processes')
    parser.add_argument('--memory', type=str, default=None, help='Memory per worker (GB)')
    parser.add_argument('--nworkers', type=int, default=4, help='Number of workers')
    parser.add_argument('--cores', type=int, default=4, help='Number of cores')
    parser.add_argument('--use_dask', type=str, default='True', help='Use dask processing?')
    # Simulation parameters
    parser.add_argument('--time_chunk', type=float, default=3600.0, help="Time for a chunk (s)")
    parser.add_argument('--mode', type=str, default='wind',
                        help="Mode of simulation: wind_pointing|random_pointing|polarisation|ionosphere|" \
                             "troposphere|heterogeneous")
    parser.add_argument('--duration', type=str, default='long',
                        help="Type of duration: long or medium or short")
    parser.add_argument('--wind_conditions', type=str, default='precision',
                        help="SKA definition of wind conditions: precision|standard|degraded")
    parser.add_argument('--global_pe', type=float, nargs=2, default=[0.0, 0.0], help='Global pointing error')
    parser.add_argument('--static_pe', type=float, nargs=2, default=[0.0, 0.0],
                        help='Multipliers for static errors')
    parser.add_argument('--dynamic_pe', type=float, default=1.0, help='Multiplier for dynamic errors')
    parser.add_argument('--pointing_directory', type=str, default=rascil_data_path('models'),
                        help='Location of wind PSD pointing files')
    parser.add_argument('--vp_directory', type=str, default=rascil_data_path('models/interpolated'),
                        help='Location of voltage pattern files')
    parser.add_argument('--show', type=str, default='False', help='Show details of simulation?')
    ### SLURM
    parser.add_argument('--use_slurm', type=str, default='False', help='Use SLURM?')
    parser.add_argument('--slurm_project', type=str, default='SKA-SDP', help='SLURM project for accounting')
    parser.add_argument('--slurm_queue', type=str, default='compute', help='SLURM queue')
    parser.add_argument('--slurm_walltime', type=str, default='01:00:00', help='SLURM time limit')

    return parser
    

if __name__ == "__main__":
    # Get command line inputs
    
    # Get command line inputs
    parser = cli_parser()

    args = parser.parse_args()

    simulation(args)
