""" Functions that help with SKA simulations

"""

__all__ = ['plot_visibility', 'plot_visibility_pol', 'find_times_above_elevation_limit', 'plot_uvcoverage',
           'plot_uwcoverage', 'plot_vwcoverage', 'plot_configuration',
           'plot_azel', 'plot_gaintable', 'plot_pointingtable', 'find_pb_width_null',
           'create_simulation_components', 'create_mid_simulation_components', 'plot_pa']

import logging

import astropy.units as units
import matplotlib.pyplot as plt
import numpy
from astropy.coordinates import SkyCoord

from rascil import phyconst
from rascil.data_models.memory_data_models import Skycomponent
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.image import create_image
from rascil.processing_components.image.operations import show_image
from rascil.processing_components.imaging.primary_beams import create_pb
from rascil.processing_components.skycomponent.base import copy_skycomponent
from rascil.processing_components.skycomponent.operations import apply_beam_to_skycomponent, \
    filter_skycomponents_by_flux
from rascil.processing_components.util.coordinate_support import hadec_to_azel
from rascil.processing_components.visibility.visibility_geometry import calculate_blockvisibility_hourangles, \
    calculate_blockvisibility_azel, calculate_blockvisibility_parallactic_angles

log = logging.getLogger('rascil-logger')


def find_times_above_elevation_limit(start_times, end_times, location, phasecentre, elevation_limit):
    """ Find all times for which a phasecentre is above the elevation limit
    
    :param start_times:
    :param end_times:
    :param location:
    :param phasecentre:
    :param elevation_limit:
    :return:
    """
    assert len(start_times) == len(end_times)
    
    def valid_elevation(time, location, phasecentre):
        ha = numpy.pi * time / 43200.0
        dec = phasecentre.dec.rad
        az, el = hadec_to_azel(ha, dec, location.lat.rad)
        return el > elevation_limit * numpy.pi / 180.0
    
    number_valid_times = 0
    valid_start_times = []
    for it, t in enumerate(start_times):
        if valid_elevation(start_times[it], location, phasecentre) or \
                valid_elevation(end_times[it], location, phasecentre):
            valid_start_times.append(t)
            number_valid_times += 1
    
    assert number_valid_times > 0, "No data above elevation limit"
    
    return valid_start_times


def plot_visibility(vis_list, colors=None, title='Visibility', y='amp', x='uvdist', plot_file=None, chan=0, **kwargs):
    """ Standard plot of visibility

    :param vis_list:
    :param plot_file:
    :param kwargs:
    :return:
    """
    plt.clf()
    if colors is None:
        colors = numpy.repeat(['b'], len(vis_list))
    
    for ivis, vis in enumerate(vis_list):
        if y == 'amp':
            yvalue = numpy.abs(vis.blockvisibility_acc.flagged_vis[..., chan, 0]).flat
        else:
            yvalue = numpy.angle(vis.blockvisibility_acc.flagged_vis[..., chan, 0]).flat
        xvalue = vis.blockvisibility_acc.uvdist.data.flat
        plt.plot(xvalue[yvalue > 0.0], yvalue[yvalue > 0.0], '.', color=colors[ivis], markersize=0.2)
    
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    if plot_file is not None:
        plt.savefig(plot_file)
    plt.show(block=False)


def plot_visibility_pol(vis_list, title='Visibility_pol', y='amp', x='uvdist', plot_file=None, chan=0, **kwargs):
    """ Standard plot of visibility

    :param vis_list:
    :param plot_file:
    :param kwargs:
    :return:
    """
    plt.clf()
    for ivis, vis in enumerate(vis_list):
        pols = vis.blockvisibility_acc.polarisation_frame.names
        colors = ["red", "blue", "green", "purple"]
        for pol in range(vis.blockvisibility_acc.npol):
            if y == 'amp':
                yvalue = numpy.abs(vis.blockvisibility_acc.flagged_vis[..., chan, pol]).flat
            else:
                yvalue = numpy.angle(vis.blockvisibility_acc.flagged_vis[..., chan, pol]).flat
            if x == "time":
                xvalue = numpy.repeat(vis["time"].data, len(yvalue))
            else:
                xvalue = vis.blockvisibility_acc.uvdist.data.flat
            if ivis == 0:
                plt.plot(xvalue[yvalue > 0.0], yvalue[yvalue > 0.0], '.', color=colors[pol],
                         label=pols[pol])
            else:
                plt.plot(xvalue[yvalue > 0.0], yvalue[yvalue > 0.0], '.', color=colors[pol])
    
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    if plot_file is not None:
        plt.savefig(plot_file)
    plt.show(block=False)


def plot_uvcoverage(vis_list, ax=None, plot_file=None, title='UV coverage', **kwargs):
    """ Standard plot of uv coverage

    :param vis_list:
    :param plot_file:
    :param kwargs:
    :return:
    """
    
    for ivis, ovis in enumerate(vis_list):
        gvis = ovis.where(ovis["flags"] == 0)
        bvis = ovis.where(ovis["flags"] > 0)
        u = numpy.array(gvis.uvw_lambda.sel(spatial='u').data.flat)
        v = numpy.array(gvis.uvw_lambda.sel(spatial='v').data.flat)
        if ivis == 0:
            plt.plot(u, v, '.', color='b', markersize=0.2, label="Unflagged")
        else:
            plt.plot(u, v, '.', color='b', markersize=0.2,)

        plt.plot(-u, -v, '.', color='b', markersize=0.2)
        u = numpy.array(bvis.uvw_lambda.sel(spatial='u').data.flat)
        v = numpy.array(bvis.uvw_lambda.sel(spatial='v').data.flat)
        if ivis == 0:
            plt.plot(u, v, '.', color='r', markersize=0.2, label="Flagged")
        else:
            plt.plot(u, v, '.', color='r', markersize=0.2)
        plt.plot(-u, -v, '.', color='r', markersize=0.2)
    plt.xlabel('U (wavelengths)')
    plt.ylabel('V (wavelengths)')
    plt.legend()
    plt.title(title)
    if plot_file is not None:
        plt.savefig(plot_file)
    plt.show(block=False)


def plot_configuration(config, ax=None, plot_file=None, title='Configuration', label=False, **kwargs):
    """ Standard plot of uv coverage

    :param vis_list:
    :param plot_file:
    :param kwargs:
    :return:
    """
    antxyz = config.xyz.data
    names = config.names.data
    if label:
        plt.plot(antxyz[:, 0], antxyz[:, 1], '.', color='b', markersize=2.4)
        for iant, name in enumerate(names):
            plt.annotate(name, (antxyz[iant, 0], antxyz[iant, 1]))
    else:
        plt.plot(antxyz[:, 0], antxyz[:, 1], '.', color='b', markersize=10.0)
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(title)
    if plot_file is not None:
        plt.savefig(plot_file)
    plt.show(block=False)


def plot_uwcoverage(vis_list, ax=None, plot_file=None, title='UW coverage', **kwargs):
    """ Standard plot of uw coverage

    :param vis_list:
    :param plot_file:
    :param kwargs:
    :return:
    """
    
    for ivis, vis in enumerate(vis_list):
        u = numpy.array(vis.blockvisibility_acc.u.data[...].flat)
        w = numpy.array(vis.blockvisibility_acc.w.data[...].flat)
        k = (vis["frequency"].data / phyconst.c_m_s)
        u = numpy.array(numpy.outer(u, k).flat)
        w = numpy.array(numpy.outer(w, k).flat)
        plt.plot(u, w, '.', color='b', markersize=0.2)
        plt.plot(-u, -w, '.', color='b', markersize=0.2)
    plt.xlabel('U (wavelengths)')
    plt.ylabel('W (wavelengths)')
    plt.title(title)
    if plot_file is not None:
        plt.savefig(plot_file)
    plt.show(block=False)


def plot_vwcoverage(vis_list, ax=None, plot_file=None, title='VW coverage', **kwargs):
    """ Standard plot of vw coverage

    :param vis_list:
    :param plot_file:
    :param kwargs:
    :return:
    """
    
    for ivis, vis in enumerate(vis_list):
        v = numpy.array(vis.blockvisibility_acc.v.data[...].flat)
        w = numpy.array(vis.blockvisibility_acc.w.data[...].flat)
        k = vis["frequency"].data / phyconst.c_m_s
        v = numpy.array(numpy.outer(v, k).flat)
        w = numpy.array(numpy.outer(w, k).flat)
        plt.plot(v, w, '.', color='b', markersize=0.2)
        plt.plot(-v, -w, '.', color='b', markersize=0.2)
    plt.xlabel('V (wavelengths)')
    plt.ylabel('W (wavelengths)')
    plt.title(title)
    if plot_file is not None:
        plt.savefig(plot_file)
    plt.show(block=False)


def plot_azel(bvis_list, plot_file=None, **kwargs):
    """ Standard plot of az el coverage
    
    :param bvis_list:
    :param plot_file:
    :param kwargs:
    :return:
    """
    plt.clf()
    r2h = 12.0 / numpy.pi
    
    for ibvis, bvis in enumerate(bvis_list):
        ha = r2h * calculate_blockvisibility_hourangles(bvis).value
        az, el = calculate_blockvisibility_azel(bvis)
        if ibvis == 0:
            plt.plot(ha, az.deg, '.', color='r', label='Azimuth (deg)')
            plt.plot(ha, el.deg, '.', color='b', label='Elevation (deg)')
        else:
            plt.plot(ha, az.deg, '.', color='r')
            plt.plot(ha, el.deg, '.', color='b')
    plt.xlabel('HA (hours)')
    plt.ylabel('Angle')
    plt.legend()
    plt.title('Azimuth and elevation vs hour angle')
    if plot_file is not None:
        plt.savefig(plot_file)
    plt.show(block=False)


def plot_pa(bvis_list, plot_file=None, **kwargs):
    """ Standard plot of parallactic angle coverage

    :param bvis_list:
    :param plot_file:
    :param kwargs:
    :return:
    """
    plt.clf()
    
    for ibvis, bvis in enumerate(bvis_list):
        ha = calculate_blockvisibility_hourangles(bvis).value
        pa = calculate_blockvisibility_parallactic_angles(bvis)
        if ibvis == 0:
            plt.plot(ha, pa.deg, '.', color='r', label='PA (deg)')
        else:
            plt.plot(ha, pa.deg, '.', color='r')
    plt.xlabel('HA (hours)')
    plt.ylabel('Parallactic Angle')
    plt.legend()
    plt.title('Parallactic angle vs hour angle')
    if plot_file is not None:
        plt.savefig(plot_file)
    plt.show(block=False)


def plot_gaintable(gt_list, title='', value='amp', plot_file=None, **kwargs):
    """ Standard plot of gain table
    
    :param gt_list:
    :param title:
    :param plot_file:
    :param kwargs:
    :return:
    """
    plt.clf()
    for igt, gt in enumerate(gt_list):
        nrec = gt[0].gaintable_acc.nrec
        names = gt[0].receptor1.data
        if nrec > 1:
            recs = [0, 1]
        else:
            recs = [1]
        
        colors = ['r', 'b']
        for irec, rec in enumerate(recs):
            amp = numpy.abs(gt[0].gain[:, 0, 0, rec, rec])
            if value == 'phase':
                y = numpy.angle(gt[0].gain[:, 0, 0, rec, rec])
                if igt == 0:
                    plt.plot(gt[0].time[amp > 0.0], y[amp > 0.0], '.', color=colors[rec], label=names[rec])
                else:
                    plt.plot(gt[0].time[amp > 0.0], y[amp > 0.0], '.', color=colors[rec])
            else:
                y = amp
                if igt == 0:
                    plt.plot(gt[0].time[amp > 0.0], 1.0 / y[amp > 0.0], '.', color=colors[rec], label=names[rec])
                else:
                    plt.plot(gt[0].time[amp > 0.0], 1.0 / y[amp > 0.0], '.', color=colors[rec])
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.legend()
    if plot_file is not None:
        plt.savefig(plot_file)
    plt.show(block=False)


def plot_pointingtable(pt_list, plot_file, title, **kwargs):
    """ Standard plot of pointing table
    
    :param pt_list:
    :param plot_file:
    :param title:
    :param kwargs:
    :return:
    """
    plt.clf()
    r2a = 180.0 * 3600.0 / numpy.pi
    rms_az = 0.0
    rms_el = 0.0
    num = 0
    for pt in pt_list:
        num += len(pt.pointing[:, 0, 0, 0, 0])
        rms_az += numpy.sum((r2a * pt.pointing[:, 0, 0, 0, 0]) ** 2)
        rms_el += numpy.sum((r2a * pt.pointing[:, 0, 0, 0, 1]) ** 2)
        plt.plot(pt.time, r2a * pt.pointing[:, 0, 0, 0, 0], '.', color='r')
        plt.plot(pt.time, r2a * pt.pointing[:, 0, 0, 0, 1], '.', color='b')
    
    rms_az = numpy.sqrt(rms_az / num)
    rms_el = numpy.sqrt(rms_el / num)
    plt.title("%s az, el rms %.2f %.2f (arcsec)" % (title, rms_az, rms_el))
    plt.xlabel('Time (s)')
    plt.ylabel('Offset (arcsec)')
    if plot_file is not None:
        plt.savefig(plot_file)
    plt.show(block=False)


def find_pb_width_null(pbtype, frequency, **kwargs):
    """ Rough estimates of HWHM and null locations
    
    :param pbtype:
    :param frequency:
    :param kwargs:
    :return:
    """
    if pbtype == 'MID':
        HWHM_deg = 0.596 * 1.36e9 / frequency[0]
        null_az_deg = 2.0 * HWHM_deg
        null_el_deg = 2.0 * HWHM_deg
    elif pbtype == 'MID_FEKO_B1':
        null_az_deg = 1.0779 * 1.36e9 / frequency[0]
        null_el_deg = 1.149 * 1.36e9 / frequency[0]
        HWHM_deg = 0.447 * 1.36e9 / frequency[0]
    elif pbtype == 'MID_FEKO_B2':
        null_az_deg = 1.0779 * 1.36e9 / frequency[0]
        null_el_deg = 1.149 * 1.36e9 / frequency[0]
        HWHM_deg = 0.447 * 1.36e9 / frequency[0]
    elif pbtype == 'MID_FEKO_Ku':
        null_az_deg = 1.0779 * 1.36e9 / frequency[0]
        null_el_deg = 1.149 * 1.36e9 / frequency[0]
        HWHM_deg = 0.447 * 1.36e9 / frequency[0]
    else:
        null_az_deg = 1.145 * 1.36e9 / frequency[0]
        null_el_deg = 1.145 * 1.36e9 / frequency[0]
        HWHM_deg = 0.447 * 1.36e9 / frequency[0]
    
    return HWHM_deg, null_az_deg, null_el_deg


def create_simulation_components(context, phasecentre, frequency, pbtype, offset_dir, flux_limit,
                                 pbradius, pb_npixel, pb_cellsize, show=False, fov=10,
                                 polarisation_frame=PolarisationFrame("stokesI"),
                                 filter_by_primary_beam=True, flux_max=10.0):
    """ Construct components for simulation
    
    :param context: singlesource or null or s3sky
    :param phasecentre: Centre of components
    :param frequency: Frequency
    :param pbtype: Type of primary beam
    :param offset_dir: Offset in ra, dec degrees
    :param flux_limit: Lower limit flux
    :param pbradius: Radius of components in radians
    :param pb_npixel: Number of pixels in the primary beam model
    :param pb_cellsize: Cellsize in primary beam model
    :param fov: FOV in degrees (used to select catalog)
    :param flux_max: Maximum flux in model before application of primary beam
    :param filter_by_primary_beam: Filter components by primary beam
    :param polarisation_frame:
    :param show:

    :return:
    """
    
    HWHM_deg, null_az_deg, null_el_deg = find_pb_width_null(pbtype, frequency)
    
    dec = phasecentre.dec.deg
    ra = phasecentre.ra.deg
    
    if context == 'singlesource':
        log.info("create_simulation_components: Constructing single component")
        offset = [HWHM_deg * offset_dir[0], HWHM_deg * offset_dir[1]]
        log.info(
            "create_simulation_components: Offset from pointing centre = %.3f, %.3f deg" % (
                offset[0], offset[1]))
        
        # The point source is offset to approximately the halfpower point
        odirection = SkyCoord(
            ra=(ra + offset[0] / numpy.cos(numpy.pi * dec / 180.0)) * units.deg,
            dec=(dec + offset[1]) * units.deg,
            frame='icrs', equinox='J2000')
        
        if polarisation_frame.type == "stokesIQUV":
            original_components = [
                Skycomponent(flux=[[1.0, 0.0, 0.0, 0.0]], direction=odirection, frequency=frequency,
                             polarisation_frame=PolarisationFrame('stokesIQUV'))]
        else:
            original_components = [
                Skycomponent(flux=[[1.0]], direction=odirection, frequency=frequency,
                             polarisation_frame=PolarisationFrame('stokesI'))]
        
        offset_direction = odirection
    
    elif context == 'doublesource':
        
        original_components = []
        
        log.info("create_simulation_components: Constructing double components")
        
        for sign_offset in [(-1, 0), (1, 0)]:
            offset = [HWHM_deg * sign_offset[0], HWHM_deg * sign_offset[1]]
            
            log.info(
                "create_simulation_components: Offset from pointing centre = %.3f, %.3f deg" % (
                    offset[0], offset[1]))
            
            odirection = SkyCoord(
                ra=(ra + offset[0] / numpy.cos(numpy.pi * dec / 180.0)) * units.deg,
                dec=(dec + offset[1]) * units.deg,
                frame='icrs', equinox='J2000')
            if polarisation_frame.type == "stokesIQUV":
                original_components.append(
                    Skycomponent(flux=[[1.0, 0.0, 0.0, 0.0]], direction=odirection, frequency=frequency,
                                 polarisation_frame=PolarisationFrame('stokesIQUV')))
            else:
                original_components.append(
                    Skycomponent(flux=[[1.0]], direction=odirection, frequency=frequency,
                                 polarisation_frame=PolarisationFrame('stokesI')))
        
        for o in original_components:
            print(o)
        
        offset_direction = odirection
    
    elif context == 'null':
        log.info("create_simulation_components: Constructing single component at the null")
        
        offset = [null_az_deg * offset_dir[0], null_el_deg * offset_dir[1]]
        HWHM = HWHM_deg * numpy.pi / 180.0
        
        log.info("create_simulation_components: Offset from pointing centre = %.3f, %.3f deg" % (offset[0], offset[1]))
        
        # The point source is offset to approximately the null point
        offset_direction = SkyCoord(ra=(ra + offset[0] / numpy.cos(numpy.pi * dec / 180.0)) * units.deg,
                                    dec=(dec + offset[1]) * units.deg,
                                    frame='icrs', equinox='J2000')
        
        if polarisation_frame.type == "stokesIQUV":
            original_components = [
                Skycomponent(flux=[[1.0, 0.0, 0.0, 0.0]], direction=offset_direction, frequency=frequency,
                             polarisation_frame=PolarisationFrame('stokesIQUV'))]
        else:
            original_components = [
                Skycomponent(flux=[[1.0]], direction=offset_direction, frequency=frequency,
                             polarisation_frame=PolarisationFrame('stokesI'))]
    
    else:
        offset = [0.0, 0.0]
        # Make a skymodel from S3
        max_flux = 0.0
        total_flux = 0.0
        log.info("create_simulation_components: Constructing s3sky components")
        from rascil.processing_components.simulation import create_test_skycomponents_from_s3
        
        all_components = create_test_skycomponents_from_s3(flux_limit=flux_limit,
                                                           phasecentre=phasecentre,
                                                           polarisation_frame=polarisation_frame,
                                                           frequency=numpy.array(frequency),
                                                           radius=pbradius,
                                                           fov=fov)
        original_components = filter_skycomponents_by_flux(all_components, flux_max=flux_max)
        log.info("create_simulation_components: %d components before filtering with primary beam" %
                 (len(original_components)))
        
        if filter_by_primary_beam:
            pbmodel = create_image(npixel=pb_npixel,
                                   cellsize=pb_cellsize,
                                   phasecentre=phasecentre,
                                   frequency=frequency,
                                   polarisation_frame=PolarisationFrame("stokesI"))
            stokesi_components = [copy_skycomponent(o) for o in original_components]
            for s in stokesi_components:
                s.flux = numpy.array([[s.flux[0, 0]]])
                s.polarisation_frame = PolarisationFrame("stokesI")
            
            pb = create_pb(pbmodel, "MID_GAUSS", pointingcentre=phasecentre, use_local=False)
            pb_applied_components = [copy_skycomponent(c) for c in stokesi_components]
            pb_applied_components = apply_beam_to_skycomponent(pb_applied_components, pb)
            filtered_components = []
            for icomp, comp in enumerate(pb_applied_components):
                if comp.flux[0, 0] > flux_limit:
                    total_flux += comp.flux[0, 0]
                    if abs(comp.flux[0, 0]) > max_flux:
                        max_flux = abs(comp.flux[0, 0])
                    filtered_components.append(original_components[icomp])
            log.info("create_simulation_components: %d components > %.6f Jy after filtering with primary beam" %
                     (len(filtered_components), flux_limit))
            log.info("create_simulation_components: Strongest components is %g (Jy)" % max_flux)
            log.info("create_simulation_components: Total flux in components is %g (Jy)" % total_flux)
            original_components = [copy_skycomponent(c) for c in filtered_components]
            if show:
                plt.clf()
                show_image(pb, components=original_components)
                plt.show(block=False)
        
        log.info("create_simulation_components: Created %d components" % len(original_components))
        # Primary beam points to the phasecentre
        offset_direction = SkyCoord(ra=ra * units.deg, dec=dec * units.deg, frame='icrs',
                                    equinox='J2000')
    
    return original_components, offset_direction


def create_mid_simulation_components(phasecentre, frequency, flux_limit, pbradius, pb_npixel, pb_cellsize, show=False,
                                     fov=10, polarisation_frame=PolarisationFrame("stokesI"), flux_max=10.0,
                                     pb_type="MID", apply_pb=True):
    """ Construct components for simulation

    :param context: singlesource or null or s3sky
    :param phasecentre: Centre of components
    :param frequency: Frequency
    :param pbtype: Type of primary beam
    :param offset_dir: Offset in ra, dec degrees
    :param flux_limit: Lower limit flux
    :param pbradius: Radius of components in radians
    :param pb_npixel: Number of pixels in the primary beam model
    :param pb_cellsize: Cellsize in primary beam model
    :param fov: FOV in degrees (used to select catalog)
    :param flux_max: Maximum flux in model before application of primary beam
    :param polarisation_frame:
    :param apply_pb: Apply the primary beam to the output components
    :param show:

    :return:
    """
    
    # Make a skymodel from S3
    max_flux = 0.0
    total_flux = 0.0
    log.info("create_simulation_components: Constructing s3sky components")
    from rascil.processing_components.simulation import create_test_skycomponents_from_s3
    
    all_components = create_test_skycomponents_from_s3(flux_limit=flux_limit,
                                                       phasecentre=phasecentre,
                                                       polarisation_frame=PolarisationFrame("stokesI"),
                                                       frequency=numpy.array(frequency),
                                                       radius=pbradius,
                                                       fov=fov)
    original_components = filter_skycomponents_by_flux(all_components, flux_max=flux_max)
    log.info("create_simulation_components: %d components before filtering with primary beam" %
             (len(original_components)))
    
    pbmodel = create_image(npixel=pb_npixel,
                           cellsize=pb_cellsize,
                           phasecentre=phasecentre,
                           frequency=frequency,
                           polarisation_frame=PolarisationFrame("stokesI"))
    
    pb = create_pb(pbmodel, pb_type, pointingcentre=phasecentre, use_local=False)
    pb_applied_components = apply_beam_to_skycomponent(original_components, pb)
    
    filtered_components = []
    filtered_pb_components = []
    reference_component = -1
    reference_flux = 0.0
    for icomp, comp in enumerate(original_components):
        if pb_applied_components[icomp].flux[0, 0] > flux_limit:
            total_flux += pb_applied_components[icomp].flux[0, 0]
            nchan, npol = comp.flux.shape
            scomp = copy_skycomponent(comp)
            npol = polarisation_frame.npol
            iflux = numpy.zeros([nchan, npol])
            iflux[:, 0] = scomp.flux[:, 0]
            scomp.flux = iflux
            scomp.polarisation_frame = polarisation_frame
            filtered_components.append(scomp)
            if scomp.flux[0, 0] > reference_flux:
                reference_component = len(filtered_components) - 1
            filtered_pb_components.append(pb_applied_components[icomp])

    log.info("create_simulation_components: %d components > %.6f Jy after filtering with primary beam" %
             (len(filtered_components), flux_limit))
    log.info("create_simulation_components: Total flux in components is %g (Jy)" % total_flux)
    if show:
        plt.clf()
        show_image(pb, components=filtered_components)
        plt.show(block=False)
    
    log.info("create_simulation_components: Created %d components" % len(filtered_components))
    
    if apply_pb:
        return filtered_pb_components, reference_component
    else:
        return filtered_components, reference_component
