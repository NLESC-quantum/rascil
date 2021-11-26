"""
Base simple visibility operations, placed here to avoid circular dependencies
"""

__all__ = [
    "vis_summary",
    "copy_visibility",
    "create_blockvisibility_from_ms",
    "create_blockvisibility_from_uvfits",
    "create_blockvisibility",
    "phaserotate_visibility",
    "export_blockvisibility_to_ms",
    "extend_blockvisibility_to_ms",
    "list_ms",
    "generate_baselines",
    "get_baseline",
]

import copy
import logging
import re

import astropy.constants as const
import numpy
import pandas
import xarray
from astropy import units as u, constants as constants
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.io import fits
from astropy.time import Time
from astropy.units import Quantity

from rascil.data_models.memory_data_models import BlockVisibility, Configuration
from rascil.data_models.polarisation import (
    PolarisationFrame,
    ReceptorFrame,
    correlate_polarisation,
)
from rascil.processing_components.util import skycoord_to_lmn
from rascil.processing_components.util import (
    xyz_to_uvw,
    uvw_to_xyz,
    hadec_to_azel,
    xyz_at_latitude,
    ecef_to_enu,
    enu_to_ecef,
    eci_to_uvw,
    enu_to_eci,
)
from rascil.processing_components.util.geometry import (
    calculate_transit_time,
    utc_to_ms_epoch,
)
from rascil.processing_components.visibility.visibility_geometry import (
    calculate_blockvisibility_transit_time,
    calculate_blockvisibility_hourangles,
    calculate_blockvisibility_azel,
)
from rascil import phyconst

log = logging.getLogger("rascil-logger")


# This convention agrees with that in the MS reader
# Note that ant2 > ant1


def generate_baselines(nant):
    """Generate mapping from antennas to baselines

    Note that we need to include autocorrelations since some input measurement sets
    may contain autocorrelations

    :param nant:
    :return:
    """
    for ant1 in range(0, nant):
        for ant2 in range(ant1, nant):
            yield ant1, ant2


def get_baseline(ant1, ant2, baselines):
    """Given the antenna numbers work out the baseline number.

    :param ant1:
    :param ant2:
    :param baselines:
    :return:
    """
    return baselines.get_loc((ant1, ant2))


def vis_summary(vis: BlockVisibility):
    """Return string summarizing the BlockVisibility

    :param vis: BlockVisibility
    :return: string
    """
    return "%d rows, %.3f GB" % (
        vis.blockvisibility_acc.nvis,
        vis.blockvisibility_acc.size(),
    )


def copy_visibility(vis: BlockVisibility, zero=False) -> BlockVisibility:
    """Copy a visibility

    Performs a deepcopy of the data array
    :param vis: BlockVisibility
    :returns: BlockVisibility

    """
    # assert isinstance(vis, BlockVisibility), vis

    newvis = copy.deepcopy(vis)
    if zero:
        newvis["vis"].data[...] = 0.0
    return newvis


def create_blockvisibility(
    config: Configuration,
    times: numpy.array,
    frequency: numpy.array,
    phasecentre: SkyCoord,
    weight: float = 1.0,
    polarisation_frame: PolarisationFrame = None,
    integration_time=1.0,
    channel_bandwidth=1e6,
    zerow=False,
    elevation_limit=15.0 * numpy.pi / 180.0,
    source="unknown",
    meta=None,
    utc_time=None,
    times_are_ha=True,
    **kwargs
) -> BlockVisibility:
    """Create a BlockVisibility from Configuration, hour angles, and direction of source

    Note that we keep track of the integration time for BDA purposes

    The input times are hour angles in radians, these are converted to UTC MJD in seconds, using utc_time as
    the approximate time.

    :param config: Configuration of antennas
    :param times: time or hour angles in radians
    :param times_are_ha: The times are hour angles (default) instead of utc time (in radians)
    :param frequency: frequencies (Hz] [nchan]
    :param weight: weight of a single sample
    :param phasecentre: phasecentre of observation (SkyCoord)
    :param channel_bandwidth: channel bandwidths: (Hz] [nchan]
    :param integration_time: Integration time ('auto' or value in s)
    :param polarisation_frame: PolarisationFrame('stokesI')
    :param integration_time: in seconds
    :param zerow: bool - set w to zero
    :param elevation_limit: in degrees
    :param source: Source name
    :param meta: Meta data as a dictionary
    :param utc_time: Time of ha definition default is Time("2000-01-01T00:00:00", format='isot', scale='utc')
    :return: BlockVisibility
    """
    assert phasecentre is not None, "Must specify phase centre"

    if utc_time is None:
        utc_time_zero = Time("2000-01-01T00:00:00", format="isot", scale="utc")
    elif isinstance(utc_time, Time):
        utc_time_zero = utc_time
        utc_time = None

    if polarisation_frame is None:
        polarisation_frame = correlate_polarisation(config.receptor_frame)

    latitude = config.location.geodetic[1].to("rad").value
    ants_xyz = config["xyz"].data
    ants_xyz = xyz_at_latitude(ants_xyz, latitude)
    nants = len(config["names"].data)

    baselines = pandas.MultiIndex.from_tuples(
        generate_baselines(nants), names=("antenna1", "antenna2")
    )
    nbaselines = len(baselines)

    # Find the number of integrations ab
    ntimes = 0
    n_flagged = 0

    for itime, time in enumerate(times):

        # Calculate the positions of the antennas as seen for this hour angle
        # and declination
        if times_are_ha:
            ha = time
        else:
            ha = time * (phyconst.sidereal_day_seconds / 86400.0)

        _, elevation = hadec_to_azel(ha, phasecentre.dec.rad, latitude)
        if elevation_limit is None or (elevation > elevation_limit):
            ntimes += 1
        else:
            n_flagged += 1

    assert ntimes > 0, "No unflagged points"

    if elevation_limit is not None and n_flagged > 0:
        log.info(
            "create_blockvisibility: flagged %d/%d times below elevation limit %f (rad)"
            % (n_flagged, ntimes, 180.0 * elevation_limit / numpy.pi)
        )
    else:
        log.debug("create_blockvisibility: created %d times" % (ntimes))

    npol = polarisation_frame.npol
    nchan = len(frequency)
    visshape = [ntimes, nbaselines, nchan, npol]
    rvis = numpy.zeros(visshape, dtype="complex")
    rflags = numpy.ones(visshape, dtype="int")
    rweight = numpy.ones(visshape)
    rimaging_weight = numpy.ones(visshape)
    rtimes = numpy.zeros([ntimes])
    rintegrationtime = numpy.zeros([ntimes])
    ruvw = numpy.zeros([ntimes, nbaselines, 3])

    if utc_time is None:
        stime = calculate_transit_time(config.location, utc_time_zero, phasecentre)
        if stime.masked:
            stime = utc_time_zero

    # Do each time filling in the actual values
    itime = 0
    for _, time in enumerate(times):

        if times_are_ha:
            ha = time
        else:
            ha = time * (phyconst.sidereal_day_seconds / 86400.0)

        # Calculate the positions of the antennas as seen for this hour angle
        # and declination
        _, elevation = hadec_to_azel(ha, phasecentre.dec.rad, latitude)
        if elevation_limit is None or (elevation > elevation_limit):
            rtimes[itime] = stime.mjd * 86400.0 + time * 86400.0 / (2.0 * numpy.pi)
            rweight[itime, ...] = 1.0
            rflags[itime, ...] = 1

            # Loop over all pairs of antennas. Note that a2>a1
            ant_pos = xyz_to_uvw(ants_xyz, ha, phasecentre.dec.rad)

            for ibaseline, (a1, a2) in enumerate(baselines):
                if a1 != a2:
                    rweight[itime, ibaseline, ...] = weight
                    rimaging_weight[itime, ibaseline, ...] = weight
                    rflags[itime, ibaseline, ...] = 0
                else:
                    rweight[itime, ibaseline, ...] = 0.0
                    rimaging_weight[itime, ibaseline, ...] = 0.0
                    rflags[itime, ibaseline, ...] = 1

                ruvw[itime, ibaseline, :] = ant_pos[a2, :] - ant_pos[a1, :]
                rflags[itime, ibaseline, ...] = 0

            if itime > 0:
                rintegrationtime[itime] = rtimes[itime] - rtimes[itime - 1]
            itime += 1

    if itime > 1:
        rintegrationtime[0] = rintegrationtime[1]
    else:
        rintegrationtime[0] = integration_time
    rchannel_bandwidth = channel_bandwidth
    if zerow:
        ruvw[..., 2] = 0.0

    vis = BlockVisibility(
        uvw=ruvw,
        time=rtimes,
        frequency=frequency,
        vis=rvis,
        weight=rweight,
        baselines=baselines,
        imaging_weight=rimaging_weight,
        flags=rflags,
        integration_time=rintegrationtime,
        channel_bandwidth=rchannel_bandwidth,
        polarisation_frame=polarisation_frame,
        source=source,
        meta=meta,
        phasecentre=phasecentre,
        configuration=config,
    )

    log.debug("create_blockvisibility: %s" % (vis_summary(vis)))

    return vis


def phaserotate_visibility(
    vis: BlockVisibility, newphasecentre: SkyCoord, tangent=True, inverse=False
) -> BlockVisibility:
    """Phase rotate from the current phase centre to a new phase centre

    If tangent is False the uvw are recomputed and the visibility phasecentre is updated.
    Otherwise only the visibility phases are adjusted

    :param vis: BlockVisibility to be rotated
    :param newphasecentre: SkyCoord of new phasecentre
    :param tangent: Stay on the same tangent plane? (True)
    :param inverse: Actually do the opposite
    :return: BlockVisibility or BlockVisibility
    """
    l, m, n = skycoord_to_lmn(newphasecentre, vis.phasecentre)

    # No significant change?
    if numpy.abs(n) < 1e-15:
        return vis

    # Make a new copy
    newvis = copy_visibility(vis)

    phasor = calculate_blockvisibility_phasor(newphasecentre, newvis)
    assert vis["vis"].data.shape == phasor.shape
    if inverse:
        newvis["vis"].data *= phasor
    else:
        newvis["vis"].data *= numpy.conj(phasor)
    # To rotate UVW, rotate into the global XYZ coordinate system and back. We have the option of
    # staying on the tangent plane or not. If we stay on the tangent then the raster will
    # join smoothly at the edges. If we change the tangent then we will have to reproject to get
    # the results on the same image, in which case overlaps or gaps are difficult to deal with.
    if not tangent:
        # The rotation can be done on the uvw (metres) values but we also have to update
        # The wavelength dependent values
        nrows, nbl, _ = vis.uvw.shape
        if inverse:
            uvw_linear = vis.uvw.data.reshape([nrows * nbl, 3])
            xyz = uvw_to_xyz(
                uvw_linear,
                ha=-newvis.phasecentre.ra.rad,
                dec=newvis.phasecentre.dec.rad,
            )
            uvw_linear = xyz_to_uvw(
                xyz, ha=-newphasecentre.ra.rad, dec=newphasecentre.dec.rad
            )[...]
        else:
            # This is the original (non-inverse) code
            uvw_linear = newvis.uvw.data.reshape([nrows * nbl, 3])
            xyz = uvw_to_xyz(
                uvw_linear,
                ha=-newvis.phasecentre.ra.rad,
                dec=newvis.phasecentre.dec.rad,
            )
            uvw_linear = xyz_to_uvw(
                xyz, ha=-newphasecentre.ra.rad, dec=newphasecentre.dec.rad
            )[...]
        newvis.attrs["phasecentre"] = newphasecentre
        newvis["uvw"].data[...] = uvw_linear.reshape([nrows, nbl, 3])
        newvis = calculate_blockvisibility_uvw_lambda(newvis)
    return newvis


def extend_blockvisibility_to_ms(msname, bvis):
    try:
        import casacore.tables.tableutil as pt
        from casacore.tables import (
            makescacoldesc,
            makearrcoldesc,
            table,
            maketabdesc,
            tableexists,
            tableiswritable,
            tableinfo,
            tablefromascii,
            tabledelete,
            makecoldesc,
            msconcat,
            removeDerivedMSCal,
            taql,
            tablerename,
            tablecopy,
            tablecolumn,
            addDerivedMSCal,
            removeImagingColumns,
            addImagingColumns,
            required_ms_desc,
            tabledefinehypercolumn,
            default_ms,
            makedminfo,
            default_ms_subtable,
        )
        from rascil.processing_components.visibility.msv2fund import Antenna, Stand
    except ModuleNotFoundError:
        raise ModuleNotFoundError("casacore is not installed")

    try:
        from rascil.processing_components.visibility import msv2
    except ModuleNotFoundError:
        raise ModuleNotFoundError("cannot import msv2")

    # Determine if file exists
    import os

    if not os.path.exists(msname):
        if bvis is not None:
            export_blockvisibility_to_ms(msname, [bvis])
    else:
        if bvis is not None:
            extend_blockvisibility_ms_row(msname, bvis)


def extend_blockvisibility_ms_row(msname, vis):
    """Minimal BlockVisibility to MS converter

    The MS format is much more general than the RASCIL BlockVisibility so we cut many corners. This requires casacore to be
    installed. If not an exception ModuleNotFoundError is raised.

    Write a list of BlockVisibility's to a MS file, split by field and spectral window

    :param msname: File name of MS
    :param vis_list: list of BlockVisibility
    :return:
    """

    try:
        import casacore.tables.tableutil as pt
        from casacore.tables import (
            makescacoldesc,
            makearrcoldesc,
            table,
            maketabdesc,
            tableexists,
            tableiswritable,
            tableinfo,
            tablefromascii,
            tabledelete,
            makecoldesc,
            msconcat,
            removeDerivedMSCal,
            taql,
            tablerename,
            tablecopy,
            tablecolumn,
            addDerivedMSCal,
            removeImagingColumns,
            addImagingColumns,
            required_ms_desc,
            tabledefinehypercolumn,
            default_ms,
            makedminfo,
            default_ms_subtable,
        )
        from rascil.processing_components.visibility.msv2fund import Antenna, Stand
    except ModuleNotFoundError:
        raise ModuleNotFoundError("casacore is not installed")

    try:
        from rascil.processing_components.visibility import msv2
    except ModuleNotFoundError:
        raise ModuleNotFoundError("cannot import msv2")

    ms_temp = msname + "____"
    export_blockvisibility_to_ms(ms_temp, [vis], source_name=None)

    try:
        t = table(msname, readonly=False, ack=False)
        log.debug("Open ms table: %s" % str(t.info()))
        tmp = table(ms_temp, readonly=True, ack=False)
        log.debug("Open ms table: %s" % str(tmp.info()))
        tmp.copyrows(t)
        log.debug("Merge  data")
        tmp.close()
        t.flush()
        t.close()
    finally:
        import os, shutil

        if os.path.exists(ms_temp):
            shutil.rmtree(ms_temp, ignore_errors=False)


def export_blockvisibility_to_ms(msname, vis_list, source_name=None):
    """Minimal BlockVisibility to MS converter

    The MS format is much more general than the RASCIL BlockVisibility so we cut many corners. This requires casacore to be
    installed. If not an exception ModuleNotFoundError is raised.

    Write a list of BlockVisibility's to a MS file, split by field and spectral window

    :param msname: File name of MS
    :param vis_list: list of BlockVisibility
    :param source_name: Source name to use
    :param ack: Ask casacore to acknowledge each table operation
    :return:
    """
    try:
        import casacore.tables.tableutil as pt
        from casacore.tables import (
            makescacoldesc,
            makearrcoldesc,
            table,
            maketabdesc,
            tableexists,
            tableiswritable,
            tableinfo,
            tablefromascii,
            tabledelete,
            makecoldesc,
            msconcat,
            removeDerivedMSCal,
            taql,
            tablerename,
            tablecopy,
            tablecolumn,
            addDerivedMSCal,
            removeImagingColumns,
            addImagingColumns,
            required_ms_desc,
            tabledefinehypercolumn,
            default_ms,
            makedminfo,
            default_ms_subtable,
        )
        from rascil.processing_components.visibility.msv2fund import Antenna, Stand
    except ModuleNotFoundError:
        raise ModuleNotFoundError("casacore is not installed")

    try:
        from rascil.processing_components.visibility import msv2
    except ModuleNotFoundError:
        raise ModuleNotFoundError("cannot import msv2")

    # log.debug("create_blockvisibility_from_ms: %s" % str(tab.info()))
    # Start the table
    tbl = msv2.Ms(msname, ref_time=0, source_name=source_name, if_delete=True)
    for vis in vis_list:
        if source_name is None:
            source_name = vis.source
        # Check polarisation
        npol = vis.blockvisibility_acc.npol
        nchan = vis.blockvisibility_acc.nchan
        if vis.blockvisibility_acc.polarisation_frame.type == "linear":
            polarization = ["XX", "XY", "YX", "YY"]
        elif vis.blockvisibility_acc.polarisation_frame.type == "linearnp":
            polarization = ["XX", "YY"]
        elif vis.blockvisibility_acc.polarisation_frame.type == "stokesI":
            polarization = ["XX"]
        elif vis.blockvisibility_acc.polarisation_frame.type == "circular":
            polarization = ["RR", "RL", "LR", "LL"]
        elif vis.blockvisibility_acc.polarisation_frame.type == "circularnp":
            polarization = ["RR", "LL"]
        elif vis.blockvisibility_acc.polarisation_frame.type == "stokesIQUV":
            polarization = ["I", "Q", "U", "V"]
        elif vis.blockvisibility_acc.polarisation_frame.type == "stokesIQ":
            polarization = ["I", "Q"]
        elif vis.blockvisibility_acc.polarisation_frame.type == "stokesIV":
            polarization = ["I", "V"]
        else:
            raise ValueError(
                "Unknown visibility polarisation %s"
                % (vis.blockvisibility_acc.polarisation_frame.type)
            )

        tbl.set_stokes(polarization)
        tbl.set_frequency(vis["frequency"].data, vis["channel_bandwidth"].data)
        n_ant = len(vis.attrs["configuration"].xyz)

        antennas = []
        names = vis.configuration.names.data
        xyz = vis.configuration.xyz.data
        for i in range(len(names)):
            antennas.append(
                Antenna(i, Stand(names[i], xyz[i, 0], xyz[i, 1], xyz[i, 2]))
            )

        # Set baselines and data
        bl_list = []

        antennas2 = antennas

        # for ant1 in range(0, nant):
        #     for ant2 in range(ant1, nant):
        #         yield ant1, ant2

        for a1 in range(0, n_ant):
            for a2 in range(a1, n_ant):
                bl_list.append((antennas[a1], antennas2[a2]))

        tbl.set_geometry(vis.configuration, antennas)

        int_time = vis["integration_time"].data
        assert vis["integration_time"].data.shape == vis["time"].data.shape
        # bv_vis = vis['vis']
        # bv_uvw = vis['uvw']
        #
        # Now easier since the BlockVisibility is baseline oriented

        for ntime, time in enumerate(vis["time"]):
            for ipol, pol in enumerate(polarization):
                if int_time[ntime] is not None:
                    tbl.add_data_set(
                        time.data,
                        int_time[ntime],
                        bl_list,
                        vis["vis"].data[ntime, ..., ipol],
                        pol=pol,
                        source=source_name,
                        phasecentre=vis.phasecentre,
                        uvw=vis["uvw"].data[ntime, :, :],
                    )
                else:
                    tbl.add_data_set(
                        time.data,
                        0,
                        bl_list,
                        vis["vis"].data[ntime, ..., ipol],
                        pol=pol,
                        source=source_name,
                        phasecentre=vis.phasecentre,
                        uvw=vis["uvw"].data[ntime, :, :],
                    )
    tbl.write()


def list_ms(msname, ack=False):
    """List sources and data descriptors in a MeasurementSet

    :param msname: File name of MS
    :param ack: Ask casacore to acknowledge each table operation
    :return: sources, data descriptors

    For example::
        print(list_ms('3C277.1_avg.ms'))
        (['1302+5748', '0319+415', '1407+284', '1252+5634', '1331+305'], [0, 1, 2, 3])
    """
    try:
        from casacore.tables import table  # pylint: disable=import-error
    except ModuleNotFoundError:
        raise ModuleNotFoundError("casacore is not installed")
    try:
        from rascil.processing_components.visibility import msv2
    except ModuleNotFoundError:
        raise ModuleNotFoundError("cannot import msv2")

    tab = table(msname, ack=ack)
    log.debug("list_ms: %s" % str(tab.info()))

    fieldtab = table("%s/FIELD" % msname, ack=False)
    sources = fieldtab.getcol("NAME")

    ddtab = table("%s/DATA_DESCRIPTION" % msname, ack=False)
    dds = list(range(ddtab.nrows()))

    return sources, dds


def create_blockvisibility_from_ms(
    msname,
    channum=None,
    start_chan=None,
    end_chan=None,
    ack=False,
    datacolumn="DATA",
    selected_sources=None,
    selected_dds=None,
    average_channels=False,
):
    """Minimal MS to BlockVisibility converter

    The MS format is much more general than the RASCIL BlockVisibility so we cut many corners.
    This requires casacore to be installed. If not an exception ModuleNotFoundError is raised.

    Creates a list of BlockVisibility's, split by field and spectral window

    Reading of a subset of channels is possible using either start_chan and end_chan or channnum. Using start_chan
    and end_chan is preferred since it only reads the channels required. Channum is more flexible and can be used to
    read a random list of channels.

    :param msname: File name of MS
    :param channum: range of channels e.g. range(17,32), default is None meaning all
    :param start_chan: Starting channel to read
    :param end_chan: End channel to read
    :param ack: Ask casacore to acknowledge each table operation
    :param datacolumn: MS data column to read DATA, CORRECTED_DATA, or MODEL_DATA
    :param selected_sources: Sources to select
    :param selected_dds: Data descriptors to select
    :param average_channels: Average all channels read
    :return: List of BlockVisibility

    For example::

        selected_sources = ['1302+5748', '1252+5634']
        bvis_list = create_blockvisibility_from_ms('../../data/3C277.1_avg.ms', datacolumn='CORRECTED_DATA',
                                           selected_sources=selected_sources)
        sources = numpy.unique([bv.source for bv in bvis_list])
        print(sources)
        ['1252+5634' '1302+5748']

    """
    try:
        from casacore.tables import table  # pylint: disable=import-error
    except ModuleNotFoundError:
        raise ModuleNotFoundError("casacore is not installed")
    try:
        from rascil.processing_components.visibility import msv2
    except ModuleNotFoundError:
        raise ModuleNotFoundError("cannot import msv2")

    tab = table(msname, ack=ack)
    log.debug("create_blockvisibility_from_ms: %s" % str(tab.info()))

    if selected_sources is None:
        fields = numpy.unique(tab.getcol("FIELD_ID"))
    else:
        fieldtab = table("%s/FIELD" % msname, ack=False)
        sources = fieldtab.getcol("NAME")
        fields = list()
        for field, source in enumerate(sources):
            if source in selected_sources:
                fields.append(field)
        assert len(fields) > 0, "No sources selected"

    if selected_dds is None:
        dds = numpy.unique(tab.getcol("DATA_DESC_ID"))
    else:
        dds = selected_dds

    log.info(
        "create_blockvisibility_from_ms: Reading unique fields %s, unique data descriptions %s"
        % (str(fields), str(dds))
    )
    vis_list = list()
    for field in fields:
        ftab = table(msname, ack=ack).query("FIELD_ID==%d" % field, style="")
        assert ftab.nrows() > 0, "Empty selection for FIELD_ID=%d" % (field)
        for dd in dds:
            # Now get info from the subtables
            ddtab = table("%s/DATA_DESCRIPTION" % msname, ack=False)
            spwid = ddtab.getcol("SPECTRAL_WINDOW_ID")[dd]
            polid = ddtab.getcol("POLARIZATION_ID")[dd]
            ddtab.close()

            meta = {"MSV2": {"FIELD_ID": field, "DATA_DESC_ID": dd}}
            ms = ftab.query("DATA_DESC_ID==%d" % dd, style="")
            assert (
                ms.nrows() > 0
            ), "Empty selection for FIELD_ID=%d and DATA_DESC_ID=%d" % (field, dd)
            log.debug("create_blockvisibility_from_ms: Found %d rows" % (ms.nrows()))
            # The TIME column has descriptor:
            # {'valueType': 'double', 'dataManagerType': 'IncrementalStMan', 'dataManagerGroup': 'TIME',
            # 'option': 0, 'maxlen': 0, 'comment': 'Modified Julian Day',
            # 'keywords': {'QuantumUnits': ['s'], 'MEASINFO': {'type': 'epoch', 'Ref': 'UTC'}}}
            otime = ms.getcol("TIME")
            datacol = ms.getcol(datacolumn, nrow=1)
            datacol_shape = list(datacol.shape)
            channels = datacol.shape[-2]
            log.debug("create_blockvisibility_from_ms: Found %d channels" % (channels))
            if channum is None:
                if start_chan is not None and end_chan is not None:
                    try:
                        log.debug(
                            "create_blockvisibility_from_ms: Reading channels from %d to %d"
                            % (start_chan, end_chan)
                        )
                        blc = [start_chan, 0]
                        trc = [end_chan, datacol_shape[-1] - 1]
                        channum = range(start_chan, end_chan + 1)
                        ms_vis = ms.getcolslice(datacolumn, blc=blc, trc=trc)
                        ms_flags = ms.getcolslice("FLAG", blc=blc, trc=trc)
                        ms_weight = ms.getcol("WEIGHT")

                    except IndexError:
                        raise IndexError("channel number exceeds max. within ms")

                else:
                    log.debug(
                        "create_blockvisibility_from_ms: Reading all %d channels"
                        % (channels)
                    )
                    try:
                        channum = range(channels)
                        ms_vis = ms.getcol(datacolumn)[:, channum, :]
                        ms_weight = ms.getcol("WEIGHT")
                        ms_flags = ms.getcol("FLAG")[:, channum, :]
                        channum = range(channels)
                    except IndexError:
                        raise IndexError("channel number exceeds max. within ms")
            else:
                log.debug(
                    "create_blockvisibility_from_ms: Reading channels %s " % (channum)
                )
                channum = range(channels)
                try:
                    ms_vis = ms.getcol(datacolumn)[:, channum, :]
                    ms_flags = ms.getcol("FLAG")[:, channum, :]
                    ms_weight = ms.getcol("WEIGHT")[:, :]
                except IndexError:
                    raise IndexError("channel number exceeds max. within ms")

            if average_channels:
                weight = ms_weight[:, numpy.newaxis, :] * (1.0 - ms_flags)
                ms_vis = numpy.sum(weight * ms_vis, axis=-2)[..., numpy.newaxis, :]
                sumwt = numpy.sum(weight, axis=-2)[..., numpy.newaxis, :]
                ms_vis[sumwt > 0.0] = ms_vis[sumwt > 0] / sumwt[sumwt > 0.0]
                ms_vis[sumwt <= 0.0] = 0.0 + 0.0j
                ms_flags = sumwt
                ms_flags[ms_flags <= 0.0] = 1.0
                ms_flags[ms_flags > 0.0] = 0.0

            uvw = -1 * ms.getcol("UVW")
            antenna1 = ms.getcol("ANTENNA1")
            antenna2 = ms.getcol("ANTENNA2")
            integration_time = ms.getcol("INTERVAL")

            time = otime - integration_time / 2.0

            start_time = numpy.min(time) / 86400.0
            end_time = numpy.max(time) / 86400.0

            log.debug(
                "create_blockvisibility_from_ms: Observation from %s to %s"
                % (Time(start_time, format="mjd").iso, Time(end_time, format="mjd").iso)
            )

            spwtab = table("%s/SPECTRAL_WINDOW" % msname, ack=False)
            cfrequency = numpy.array(spwtab.getcol("CHAN_FREQ")[spwid][channum])
            cchannel_bandwidth = numpy.array(
                spwtab.getcol("CHAN_WIDTH")[spwid][channum]
            )
            nchan = cfrequency.shape[0]
            if average_channels:
                cfrequency = numpy.array([numpy.average(cfrequency)])
                cchannel_bandwidth = numpy.array([numpy.sum(cchannel_bandwidth)])
                nchan = cfrequency.shape[0]

            # Get polarisation info
            poltab = table("%s/POLARIZATION" % msname, ack=False)
            corr_type = poltab.getcol("CORR_TYPE")[polid]
            corr_type = sorted(corr_type)
            # These correspond to the CASA Stokes enumerations
            if numpy.array_equal(corr_type, [1, 2, 3, 4]):
                polarisation_frame = PolarisationFrame("stokesIQUV")
                npol = 4
            elif numpy.array_equal(corr_type, [1, 2]):
                polarisation_frame = PolarisationFrame("stokesIQ")
                npol = 2
            elif numpy.array_equal(corr_type, [1, 4]):
                polarisation_frame = PolarisationFrame("stokesIV")
                npol = 2
            elif numpy.array_equal(corr_type, [5, 6, 7, 8]):
                polarisation_frame = PolarisationFrame("circular")
                npol = 4
            elif numpy.array_equal(corr_type, [5, 8]):
                polarisation_frame = PolarisationFrame("circularnp")
                npol = 2
            elif numpy.array_equal(corr_type, [9, 10, 11, 12]):
                polarisation_frame = PolarisationFrame("linear")
                npol = 4
            elif numpy.array_equal(corr_type, [9, 12]):
                polarisation_frame = PolarisationFrame("linearnp")
                npol = 2
            elif numpy.array_equal(corr_type, [9]):
                npol = 1
                polarisation_frame = PolarisationFrame("stokesI")
            else:
                raise KeyError("Polarisation not understood: %s" % str(corr_type))

            # Get configuration
            anttab = table("%s/ANTENNA" % msname, ack=False)
            names = numpy.array(anttab.getcol("NAME"))

            ant_map = list()
            actual = 0
            # This assumes that the names are actually filled in!
            for i, name in enumerate(names):
                if name != "":
                    ant_map.append(actual)
                    actual += 1
                else:
                    ant_map.append(-1)
            # assert actual > 0, "Dish/station names are all blank - cannot load"
            if actual == 0:
                ant_map = list(range(len(names)))
                names = numpy.repeat("No name", len(names))

            mount = numpy.array(anttab.getcol("MOUNT"))[names != ""]
            # log.info("mount is: %s" % (mount))
            diameter = numpy.array(anttab.getcol("DISH_DIAMETER"))[names != ""]
            xyz = numpy.array(anttab.getcol("POSITION"))[names != ""]
            offset = numpy.array(anttab.getcol("OFFSET"))[names != ""]
            stations = numpy.array(anttab.getcol("STATION"))[names != ""]
            names = numpy.array(anttab.getcol("NAME"))[names != ""]
            nants = len(names)

            antenna1 = list(map(lambda i: ant_map[i], antenna1))
            antenna2 = list(map(lambda i: ant_map[i], antenna2))

            baselines = pandas.MultiIndex.from_tuples(
                generate_baselines(nants), names=("antenna1", "antenna2")
            )
            nbaselines = len(baselines)

            location = EarthLocation(
                x=Quantity(xyz[0][0], "m"),
                y=Quantity(xyz[0][1], "m"),
                z=Quantity(xyz[0][2], "m"),
            )

            configuration = Configuration(
                name="",
                location=location,
                names=names,
                xyz=xyz,
                mount=mount,
                frame="geocentric",
                receptor_frame=ReceptorFrame("linear"),
                diameter=diameter,
                offset=offset,
                stations=stations,
            )
            # Get phasecentres
            fieldtab = table("%s/FIELD" % msname, ack=False)
            pc = fieldtab.getcol("PHASE_DIR")[field, 0, :]
            source = fieldtab.getcol("NAME")[field]
            phasecentre = SkyCoord(
                ra=pc[0] * u.rad, dec=pc[1] * u.rad, frame="icrs", equinox="J2000"
            )

            time_index_row = numpy.zeros_like(time, dtype="int")
            time_last = time[0]
            time_index = 0
            for row, _ in enumerate(time):
                if time[row] > time_last + 0.5 * integration_time[row]:
                    assert (
                        time[row] > time_last
                    ), "MS is not time-sorted - cannot convert"
                    time_index += 1
                    time_last = time[row]
                time_index_row[row] = time_index

            ntimes = time_index + 1

            assert ntimes == len(
                numpy.unique(time_index_row)
            ), "Error in finding data times"

            bv_times = numpy.zeros([ntimes])
            bv_vis = numpy.zeros([ntimes, nbaselines, nchan, npol]).astype("complex")
            bv_flags = numpy.zeros([ntimes, nbaselines, nchan, npol]).astype("int")
            bv_weight = numpy.zeros([ntimes, nbaselines, nchan, npol])
            bv_imaging_weight = numpy.zeros([ntimes, nbaselines, nchan, npol])
            bv_uvw = numpy.zeros([ntimes, nbaselines, 3])
            bv_integration_time = numpy.zeros([ntimes])

            for row, _ in enumerate(time):
                ibaseline = baselines.get_loc((antenna1[row], antenna2[row]))
                time_index = time_index_row[row]
                bv_times[time_index] = time[row]
                bv_vis[time_index, ibaseline, ...] = ms_vis[row, ...]
                bv_flags[time_index, ibaseline, ...][
                    ms_flags[row, ...].astype("bool")
                ] = 1
                bv_weight[time_index, ibaseline, :, ...] = ms_weight[
                    row, numpy.newaxis, ...
                ]
                bv_imaging_weight[time_index, ibaseline, :, ...] = ms_weight[
                    row, numpy.newaxis, ...
                ]
                bv_uvw[time_index, ibaseline, :] = uvw[row, :]
                bv_integration_time[time_index] = integration_time[row]

            vis_list.append(
                BlockVisibility(
                    uvw=bv_uvw,
                    baselines=baselines,
                    time=bv_times,
                    frequency=cfrequency,
                    channel_bandwidth=cchannel_bandwidth,
                    vis=bv_vis,
                    flags=bv_flags,
                    weight=bv_weight,
                    integration_time=bv_integration_time,
                    imaging_weight=bv_imaging_weight,
                    configuration=configuration,
                    phasecentre=phasecentre,
                    polarisation_frame=polarisation_frame,
                    source=source,
                    meta=meta,
                )
            )
        tab.close()
    return vis_list


def create_blockvisibility_from_uvfits(fitsname, channum=None, ack=False, antnum=None):
    """Minimal UVFIT to BlockVisibility converter

    The UVFITS format is much more general than the RASCIL BlockVisibility so we cut many corners.

    Creates a list of BlockVisibility's, split by field and spectral window

    :param fitsname: File name of UVFITS
    :param channum: range of channels e.g. range(17,32), default is None meaning all
    :param antnum: the number of antenna
    :return:
    """

    def find_time_slots(times):
        """Find the time slots

        :param times:
        :return:
        """
        intervals = times[1:] - times[0:-1]
        integration_time = numpy.median(intervals[intervals > 0.0])
        last_time = times[0]
        time_slots = list()
        for t in times:
            if t > last_time + integration_time:
                last_time = t
                time_slots.append(last_time)

        time_slots = numpy.array(time_slots)

        return time_slots

    def param_dict(hdul):
        "Return the dictionary of the random parameters"

        """
        The keys of the dictionary are the parameter names uppercased for
        consistency. The values are the column numbers.

        If multiple parameters have the same name (e.g., DATE) their
        columns are entered as a list.
        """

        pre = re.compile(r"PTYPE(?P<i>\d+)")
        res = {}
        for k, v in hdul.header.items():
            m = pre.match(k)
            if m:
                vu = v.upper()
                if vu in res:
                    res[vu] = [res[vu], int(m.group("i"))]
                else:
                    res[vu] = int(m.group("i"))
        return res

    # Open the file
    with fits.open(fitsname) as hdul:

        # Read Spectral Window
        nspw = hdul[0].header["NAXIS5"]
        # Read Channel and Frequency Interval
        freq_ref = hdul[0].header["CRVAL4"]
        delt_freq = hdul[0].header["CDELT4"]
        # Real the number of channels in one spectral window
        channels = hdul[0].header["NAXIS4"]
        freq = numpy.zeros([nspw, channels])
        # Read Frequency or IF
        freqhdulname = "AIPS FQ"
        sdhu = hdul.index_of(freqhdulname)
        if_freq = hdul[sdhu].data["IF FREQ"].ravel()
        for i in range(nspw):
            temp = numpy.array(
                [if_freq[i] + freq_ref + delt_freq * ff for ff in range(channels)]
            )
            freq[i, :] = temp[:]
        freq_delt = numpy.ones(channels) * delt_freq
        if channum is None:
            channum = range(channels)

        # Read time. We are trying to find a discrete set of times to use in
        # BlockVisibility.
        bvtimes = Time(hdul[0].data["DATE"], hdul[0].data["_DATE"], format="jd")
        bv_times = find_time_slots(bvtimes.jd)

        ntimes = len(bv_times)

        # # Get Antenna
        # blin = hdul[0].data['BASELINE']
        antennahdulname = "AIPS AN"
        adhu = hdul.index_of(antennahdulname)
        try:
            antenna_name = hdul[adhu].data["ANNAME"]
            antenna_name = antenna_name.encode("ascii", "ignore")
        except ValueError:
            antenna_name = None

        antenna_xyz = hdul[adhu].data["STABXYZ"]
        antenna_mount = hdul[adhu].data["MNTSTA"]
        antenna_offset = hdul[adhu].data["STAXOF"]
        try:
            antenna_diameter = hdul[adhu].data["DIAMETER"]
        except (ValueError, KeyError):
            antenna_diameter = None
        # To reading some UVFITS with wrong numbers of antenna
        if antnum is not None and antenna_name is not None:
            antenna_name = antenna_name[:antnum]
            antenna_xyz = antenna_xyz[:antnum]
            antenna_mount = antenna_mount[:antnum]
            antenna_offset = antenna_offset[:antnum]
            if antenna_diameter is not None:
                antenna_diameter = antenna_diameter[:antnum]

        nants = len(antenna_xyz)

        baselines = pandas.MultiIndex.from_tuples(
            generate_baselines(nants), names=("antenna1", "antenna2")
        )
        nbaselines = len(baselines)

        # Put offset into same shape as for MS
        antenna_offset = numpy.c_[
            antenna_offset, numpy.zeros(nants), numpy.zeros(nants)
        ]

        # Get polarisation info
        npol = hdul[0].header["NAXIS3"]
        corr_type = numpy.arange(hdul[0].header["NAXIS3"]) - (
            hdul[0].header["CRPIX3"] - 1
        )
        corr_type *= hdul[0].header["CDELT3"]
        corr_type += hdul[0].header["CRVAL3"]
        # xx yy xy yx
        # These correspond to the CASA Stokes enumerations
        if numpy.array_equal(corr_type, [1, 2, 3, 4]):
            polarisation_frame = PolarisationFrame("stokesIQUV")
        elif numpy.array_equal(corr_type, [1, 4]):
            polarisation_frame = PolarisationFrame("stokesIV")
        elif numpy.array_equal(corr_type, [1, 2]):
            polarisation_frame = PolarisationFrame("stokesIQ")
        elif numpy.array_equal(corr_type, [-1, -2, -3, -4]):
            polarisation_frame = PolarisationFrame("circular")
        elif numpy.array_equal(corr_type, [-1, -4]):
            polarisation_frame = PolarisationFrame("circularnp")
        elif numpy.array_equal(corr_type, [-5, -6, -7, -8]):
            polarisation_frame = PolarisationFrame("linear")
        elif numpy.array_equal(corr_type, [-5, -8]):
            polarisation_frame = PolarisationFrame("linearnp")
        else:
            raise KeyError("Polarisation not understood: %s" % str(corr_type))

        configuration = Configuration(
            name="",
            location=None,
            names=antenna_name,
            xyz=antenna_xyz,
            mount=antenna_mount,
            frame=None,
            receptor_frame=polarisation_frame,
            diameter=antenna_diameter,
            offset=antenna_offset,
            stations=antenna_name,
        )

        # Get RA and DEC
        phase_center_ra_degrees = float(hdul[0].header["CRVAL6"])
        phase_center_dec_degrees = float(hdul[0].header["CRVAL7"])

        # Get phasecentres
        phasecentre = SkyCoord(
            ra=phase_center_ra_degrees * u.deg,
            dec=phase_center_dec_degrees * u.deg,
            frame="icrs",
            equinox="J2000",
        )

        # Get UVW
        d = param_dict(hdul[0])
        if "UU" in d:
            uu = hdul[0].data["UU"]
            vv = hdul[0].data["VV"]
            ww = hdul[0].data["WW"]
        else:
            uu = hdul[0].data["UU---SIN"]
            vv = hdul[0].data["VV---SIN"]
            ww = hdul[0].data["WW---SIN"]
        _vis = hdul[0].data["DATA"]

        row = 0
        nchan = len(channum)
        vis_list = list()
        for spw_index in range(nspw):
            bv_vis = numpy.zeros([ntimes, nbaselines, nchan, npol]).astype("complex")
            bv_flags = numpy.zeros([ntimes, nbaselines, nchan, npol]).astype("int")
            bv_weight = numpy.zeros([ntimes, nbaselines, nchan, npol])
            bv_uvw = numpy.zeros([ntimes, nbaselines, 3])
            for time_index, time in enumerate(bv_times):
                for antenna1 in range(nants - 1):
                    for antenna2 in range(antenna1, nants):
                        ibaseline = baselines.get_loc((antenna1, antenna2))
                        for channel_no, channel_index in enumerate(channum):
                            for pol_index in range(npol):
                                bv_vis[
                                    time_index, ibaseline, channel_no, pol_index
                                ] = complex(
                                    _vis[
                                        row,
                                        :,
                                        :,
                                        spw_index,
                                        channel_index,
                                        pol_index,
                                        0,
                                    ],
                                    _vis[
                                        row,
                                        :,
                                        :,
                                        spw_index,
                                        channel_index,
                                        pol_index,
                                        1,
                                    ],
                                )
                                bv_weight[
                                    time_index, ibaseline, channel_no, pol_index
                                ] = _vis[
                                    row, :, :, spw_index, channel_index, pol_index, 2
                                ]
                        bv_uvw[time_index, ibaseline, 0] = uu[row] * phyconst.c_m_s
                        bv_uvw[time_index, ibaseline, 1] = vv[row] * phyconst.c_m_s
                        bv_uvw[time_index, ibaseline, 2] = ww[row] * phyconst.c_m_s
                        row += 1

            # Convert negative weights to flags
            bv_flags[bv_weight < 0.0] = 1
            bv_weight[bv_weight < 0.0] = 0.0

            vis_list.append(
                BlockVisibility(
                    uvw=bv_uvw,
                    time=bv_times,
                    baselines=baselines,
                    frequency=freq[spw_index][channum],
                    channel_bandwidth=freq_delt[channum],
                    vis=bv_vis,
                    flags=bv_flags,
                    weight=bv_weight,
                    imaging_weight=bv_weight,
                    configuration=configuration,
                    phasecentre=phasecentre,
                    polarisation_frame=polarisation_frame,
                )
            )
    return vis_list


def calculate_blockvisibility_phasor(direction, vis):
    """Calculate the phasor for a component for a BlockVisibility

    :param comp:
    :param vis:
    :return:
    """
    # assert isinstance(vis, BlockVisibility)
    ntimes, nbaseline, nchan, npol = vis["vis"].data.shape
    l, m, n = skycoord_to_lmn(direction, vis.phasecentre)
    s = numpy.array([l, m, numpy.sqrt(1 - l ** 2 - m ** 2) - 1.0])

    phasor = numpy.ones([ntimes, nbaseline, nchan, npol], dtype="complex")
    phasor[...] = numpy.exp(
        -2j * numpy.pi * numpy.einsum("tbfs,s->tbf", vis.uvw_lambda.data, s)
    )[..., numpy.newaxis]
    return phasor


def calculate_blockvisibility_uvw_lambda(vis):
    """Recalculate the uvw_lambda values

    :param vis:
    :return:
    """
    k = vis.frequency.data / phyconst.c_m_s
    vis.uvw_lambda.data = numpy.einsum("tbs,k->tbks", vis.uvw.data, k)
    return vis
