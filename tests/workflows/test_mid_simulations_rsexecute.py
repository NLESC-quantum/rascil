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
import sys
import unittest
from functools import partial

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models import PolarisationFrame, SkyModel
from rascil.data_models.parameters import rascil_path, rascil_data_path
from rascil.processing_components.image.operations import qa_image, export_image_to_fits, import_image_from_fits
from rascil.processing_components.imaging.base import create_image_from_visibility, advise_wide_field
from rascil.processing_components.imaging.primary_beams import create_vp
from rascil.processing_components.simulation.simulation_helpers import find_pb_width_null, create_simulation_components
from rascil.processing_components.visibility import copy_visibility, export_blockvisibility_to_ms, \
    concatenate_visibility
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute
from rascil.workflows.rsexecute.imaging.imaging_rsexecute import weight_list_rsexecute_workflow, \
    sum_predict_results_rsexecute, subtract_list_rsexecute_workflow, invert_list_rsexecute_workflow
from rascil.workflows.rsexecute.simulation.simulation_rsexecute import \
    create_surface_errors_gaintable_rsexecute_workflow, \
    create_pointing_errors_gaintable_rsexecute_workflow, create_standard_mid_simulation_rsexecute_workflow, \
    create_polarisation_gaintable_rsexecute_workflow, create_heterogeneous_gaintable_rsexecute_workflow, \
    create_atmospheric_errors_gaintable_rsexecute_workflow
from rascil.workflows.rsexecute.skymodel.skymodel_rsexecute import predict_skymodel_list_rsexecute_workflow

results_dir = rascil_path('test_results')

log = logging.getLogger("rascil-logger")
log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)


class TestMIDSimulations(unittest.TestCase):
    
    def setUp(self) -> None:
        rsexecute.set_client(use_dask=True)
        self.persist = os.getenv("RASCIL_PERSIST", False)
    
    def simulation(self, args, mode='wind_pointing', band='B2',
                   image_polarisation_frame=PolarisationFrame("stokesIQUV"),
                   vis_polarisation_frame=PolarisationFrame("linear")):
        
        context = args.context
        ra = args.ra
        declination = args.declination
        integration_time = args.integration_time
        time_range = args.time_range
        time_chunk = args.time_chunk
        offset_dir = [1.0, 0.0]
        pbtype = args.pbtype
        pbradius = args.pbradius
        rmax = args.rmax
        flux_limit = args.flux_limit
        npixel = 1024
        vp_directory = args.vp_directory
        
        # Simulation specific parameters
        global_pe = numpy.array(args.global_pe)
        static_pe = numpy.array(args.static_pe)
        dynamic_pe = args.dynamic_pe
        
        seed = args.seed
        basename = os.path.basename(os.getcwd())
        
        # Set up details of simulated observation
        nfreqwin = 1
        if band == 'B1':
            frequency = [0.765e9]
        elif band == 'B2':
            frequency = [1.36e9]
        elif band == 'Ku':
            frequency = [12.179e9]
        else:
            raise ValueError("Unknown band %s" % band)
        
        phasecentre = SkyCoord(ra=ra * u.deg, dec=declination * u.deg, frame='icrs', equinox='J2000')
        
        bvis_list = create_standard_mid_simulation_rsexecute_workflow(band, rmax, phasecentre, time_range, time_chunk,
                                                                      integration_time,
                                                                      polarisation_frame=vis_polarisation_frame)
        bvis_list = rsexecute.persist(bvis_list)
        
        # We need the HWHM of the primary beam, and the location of the nulls
        HWHM_deg, null_az_deg, null_el_deg = find_pb_width_null(pbtype, frequency)
        
        HWHM = HWHM_deg * numpy.pi / 180.0
        
        FOV_deg = 8.0 * 1.36e9 / frequency[0]
        
        advice_list = rsexecute.execute(advise_wide_field)(bvis_list[0], guard_band_image=1.0, delA=0.02,
                                                           verbose=False)
        advice = rsexecute.compute(advice_list, sync=True)
        pb_npixel = 256
        d2r = numpy.pi / 180.0
        pb_cellsize = d2r * FOV_deg / pb_npixel
        cellsize = advice['cellsize']
        
        # Now construct the components
        original_components, offset_direction = create_simulation_components(context, phasecentre, frequency,
                                                                             pbtype, offset_dir, flux_limit,
                                                                             pbradius * HWHM, pb_npixel, pb_cellsize,
                                                                             polarisation_frame=image_polarisation_frame,
                                                                             filter_by_primary_beam=True)
        
        log.info("There are {} components".format(len(original_components)))
        
        vp_list = [rsexecute.execute(create_image_from_visibility)(bv, npixel=pb_npixel, frequency=frequency,
                                                                   nchan=nfreqwin, cellsize=pb_cellsize,
                                                                   phasecentre=phasecentre,
                                                                   polarisation_frame=vis_polarisation_frame,
                                                                   override_cellsize=False) for bv in bvis_list]
        vp_list = [rsexecute.execute(create_vp)(vp, pbtype, pointingcentre=phasecentre)
                   for vp in vp_list]
        future_vp_list = rsexecute.persist(vp_list)
        
        a2r = numpy.pi / (3600.0 * 1800)
        
        def get_vp(telescope, vp_directory):
            return create_vp(telescope=telescope)
        
        if mode == 'random_pointing':
            # Random pointing errors
            global_pointing_error = global_pe
            static_pointing_error = static_pe
            pointing_error = dynamic_pe
            
            no_error_gtl, error_gtl = \
                create_pointing_errors_gaintable_rsexecute_workflow(bvis_list, original_components,
                                                                    sub_vp_list=future_vp_list,
                                                                    pointing_error=a2r * pointing_error,
                                                                    static_pointing_error=a2r * static_pointing_error,
                                                                    global_pointing_error=a2r * global_pointing_error,
                                                                    seed=seed,
                                                                    show=False, basename=basename)
        elif mode == 'wind_pointing':
            # Wind-induced pointing errors
            no_error_gtl, error_gtl = \
                create_pointing_errors_gaintable_rsexecute_workflow(bvis_list, original_components,
                                                                    sub_vp_list=future_vp_list,
                                                                    time_series="wind",
                                                                    time_series_type="precision",
                                                                    seed=seed,
                                                                    show=False, basename=basename)
        elif mode == 'troposphere':
            screen = import_image_from_fits(args.screen)
            no_error_gtl, error_gtl = \
                create_atmospheric_errors_gaintable_rsexecute_workflow(bvis_list,
                                                                       original_components,
                                                                       r0=args.r0,
                                                                       screen=screen,
                                                                       height=args.height,
                                                                       type_atmosphere=args.mode,
                                                                       show=args.show == "True",
                                                                       basename=mode)
        elif mode == 'ionosphere':
            screen = import_image_from_fits(args.screen)
            no_error_gtl, error_gtl = \
                create_atmospheric_errors_gaintable_rsexecute_workflow(bvis_list,
                                                                       original_components,
                                                                       r0=args.r0,
                                                                       screen=screen,
                                                                       height=args.height,
                                                                       type_atmosphere=args.mode,
                                                                       show=args.show == "True",
                                                                       basename=mode)
        
        elif mode == 'surface':
            # Dish surface sag due to gravity
            no_error_gtl, error_gtl = \
                create_surface_errors_gaintable_rsexecute_workflow(band, bvis_list, original_components,
                                                                   vp_directory=vp_directory,
                                                                   show=False, basename=basename)
        elif mode == 'heterogeneous':
            # Different antennas
            no_error_gtl, error_gtl = \
                create_heterogeneous_gaintable_rsexecute_workflow(band, bvis_list, original_components,
                                                                  get_vp=partial(get_vp, vp_directory=""),
                                                                  show=False, basename=basename)
        elif mode == 'polarisation':
            # Polarised beams
            no_error_gtl, error_gtl = \
                create_polarisation_gaintable_rsexecute_workflow(band, bvis_list, original_components,
                                                                 get_vp=partial(get_vp, vp_directory=""),
                                                                 basename=basename,
                                                                 show=True)
        else:
            raise ValueError("Unknown type of error %s" % mode)
        
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
            [sum_predict_results_rsexecute(
                predict_skymodel_list_rsexecute_workflow(bvis, error_sm_list[ibvis], context='2d', docal=True))
             for ibvis, bvis in enumerate(error_bvis_list)]
        
        no_error_bvis_list = [rsexecute.execute(copy_visibility)(bvis, zero=True) for bvis in bvis_list]
        no_error_bvis_list = \
            [sum_predict_results_rsexecute(
                predict_skymodel_list_rsexecute_workflow(bvis, no_error_sm_list[ibvis], context='2d', docal=True))
             for ibvis, bvis in enumerate(no_error_bvis_list)]
        
        error_bvis = rsexecute.execute(concatenate_visibility, nout=1)(error_bvis_list)
        no_error_bvis = rsexecute.execute(concatenate_visibility, nout=1)(no_error_bvis_list)
        difference_bvis = subtract_list_rsexecute_workflow([error_bvis], [no_error_bvis])
        difference_bvis = rsexecute.execute(concatenate_visibility)(difference_bvis)
        
        # Perform uniform weighting
        model_list = [rsexecute.execute(create_image_from_visibility)(difference_bvis, npixel=npixel,
                                                                      frequency=frequency,
                                                                      nchan=nfreqwin, cellsize=cellsize,
                                                                      phasecentre=offset_direction,
                                                                      polarisation_frame=image_polarisation_frame)]
        
        bvis_list = weight_list_rsexecute_workflow([difference_bvis], model_list)
        bvis_list = rsexecute.compute(bvis_list, sync=True)
        
        # Now make all the residual images
        # Make one image per component
        result = invert_list_rsexecute_workflow(bvis_list, model_list, context='2d')
        
        # Actually compute the graph assembled above
        error_dirty, sumwt = rsexecute.compute(result[0], sync=True)
        
        if self.persist:
            export_image_to_fits(error_dirty,
                                 "{}/test_mid_simulations_{}_dirty.fits".format(rascil_path("test_results"), mode))
            export_blockvisibility_to_ms("{}/test_mid_simulations_{}_difference.ms".format(rascil_path("test_results"),
                                                                                           mode), bvis_list)
        return error_dirty, sumwt
    
    def get_args(self):
        
        import argparse
        
        parser = argparse.ArgumentParser(description='Simulate SKA-MID direction dependent errors')
        
        parser.add_argument('--context', type=str, default='s3sky', help='s3sky or singlesource or null')
        
        # Observation definition
        parser.add_argument('--ra', type=float, default=0.0, help='Right ascension (degrees)')
        parser.add_argument('--declination', type=float, default=-40.0, help='Declination (degrees)')
        parser.add_argument('--rmax', type=float, default=1e3, help='Maximum distance of station from centre (m)')
        parser.add_argument('--band', type=str, default='B2', help="Band")
        parser.add_argument('--integration_time', type=float, default=3600, help='Integration time (s)')
        parser.add_argument('--time_range', type=float, nargs=2, default=[-4.0, 4.0], help='Time range in hour angle')
        parser.add_argument('--image_pol', type=str, default='stokesIQUV', help='RASCIL polarisation frame for image')
        parser.add_argument('--vis_pol', type=str, default='linear',
                            help='RASCIL polarisation frame for visibility')
        
        parser.add_argument('--pbradius', type=float, default=1.5, help='Radius of sources to include (in HWHM)')
        parser.add_argument('--pbtype', type=str, default='MID', help='Primary beam model: MID or MID_GAUSS')
        parser.add_argument('--seed', type=int, default=18051955, help='Random number seed')
        parser.add_argument('--flux_limit', type=float, default=0.01, help='Flux limit (Jy)')
        
        # Control parameters
        parser.add_argument('--shared_directory', type=str, default=rascil_data_path('configurations'),
                            help='Location of configuration files')
        parser.add_argument('--results', type=str, default='./', help='Directory for results')
        
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
        
        args = parser.parse_args([])
        return args
    
    def test_wind(self):
        
        args = self.get_args()
        args.fluxlimit = 0.1
        
        error_dirty, sumwt = self.simulation(args, 'wind_pointing',
                                             image_polarisation_frame=PolarisationFrame("stokesIQUV"),
                                             vis_polarisation_frame=PolarisationFrame("linear"))
        
        qa = qa_image(error_dirty)

        # shape: '(1, 4, 1024, 1024)'
        # max: '0.0002463188957351486'
        # min: '-0.00044142488572784577'
        # maxabs: '0.00044142488572784577'
        # rms: '1.8021367426909772e-05'
        # sum: '0.01721111235515721'
        # medianabs: '0.0'
        # medianabsdevmedian: '0.0'
        # median: '0.0'

        numpy.testing.assert_almost_equal(qa.data['max'], 0.0002463188957351486, 5, err_msg=str(qa))
        numpy.testing.assert_almost_equal(qa.data['min'], -0.00044142488572784577, 5, err_msg=str(qa))
        numpy.testing.assert_almost_equal(qa.data['rms'], 1.8021367426909772e-05, 5, err_msg=str(qa))
    
    def test_heterogeneous(self):
        
        args = self.get_args()
        args.fluxlimit = 0.1
        
        error_dirty, sumwt = self.simulation(args, 'heterogeneous',
                                             image_polarisation_frame=PolarisationFrame("stokesIQUV"),
                                             vis_polarisation_frame=PolarisationFrame("linear"))
        
        qa = qa_image(error_dirty)
        
        numpy.testing.assert_almost_equal(qa.data['max'], 0.006232996309120276, 5, err_msg=str(qa))
        numpy.testing.assert_almost_equal(qa.data['min'], -0.00038496045275951873, 5, err_msg=str(qa))
        numpy.testing.assert_almost_equal(qa.data['rms'], 3.728425607449823e-05, 5, err_msg=str(qa))
    
    def test_random(self):
        
        args = self.get_args()
        args.fluxlimit = 0.1
        
        error_dirty, sumwt = self.simulation(args, 'random_pointing',
                                             image_polarisation_frame=PolarisationFrame("stokesIQUV"),
                                             vis_polarisation_frame=PolarisationFrame("linear"))
        
        qa = qa_image(error_dirty)

        # shape: '(1, 4, 1024, 1024)'
        # max: '2.472368365158522e-06'
        # min: '-9.762213174928972e-07'
        # maxabs: '2.472368365158522e-06'
        # rms: '7.831098142486326e-08'
        # sum: '1.373421891787825e-05'
        # medianabs: '0.0'
        # medianabsdevmedian: '0.0'
        # median: '0.0'

        numpy.testing.assert_almost_equal(qa.data['max'], 2.472368365158522e-06, 5, err_msg=str(qa))
        numpy.testing.assert_almost_equal(qa.data['min'], -9.762213174928972e-07, 5, err_msg=str(qa))
        numpy.testing.assert_almost_equal(qa.data['rms'], 7.831098142486326e-08, 5, err_msg=str(qa))
    
    def test_surface(self):
        
        args = self.get_args()
        args.fluxlimit = 0.1
        
        if os.path.isdir(rascil_path('models/interpolated')):
            error_dirty, sumwt = self.simulation(args, 'surface',
                                                 image_polarisation_frame=PolarisationFrame("stokesIQUV"),
                                                 vis_polarisation_frame=PolarisationFrame("linear"))
            
            qa = qa_image(error_dirty)
            
            numpy.testing.assert_almost_equal(qa.data['max'], 2.2055849698035616e-06, 5, err_msg=str(qa))
            numpy.testing.assert_almost_equal(qa.data['min'], -6.838117387793031e-07, 5, err_msg=str(qa))
            numpy.testing.assert_almost_equal(qa.data['rms'], 3.7224203394509413e-07, 5, err_msg=str(qa))
    
    def test_polarisation(self):
        
        args = self.get_args()
        args.fluxlimit = 3.0
        args.integration_time = 1800.0
        
        error_dirty, sumwt = self.simulation(args, 'polarisation',
                                             image_polarisation_frame=PolarisationFrame("stokesIQUV"),
                                             vis_polarisation_frame=PolarisationFrame("linear"))
        qa = qa_image(error_dirty)
        
        numpy.testing.assert_almost_equal(qa.data['max'], 0.0004365409779069447, 5, err_msg=str(qa))
        numpy.testing.assert_almost_equal(qa.data['min'], -0.00046135977866455243, 5, err_msg=str(qa))
        numpy.testing.assert_almost_equal(qa.data['rms'], 9.279652696332783e-06, 5, err_msg=str(qa))
