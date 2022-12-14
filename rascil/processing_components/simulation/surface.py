""" Functions for dish surface modeling


"""

__all__ = [
    "simulate_gaintable_from_zernikes",
    "simulate_gaintable_from_voltage_pattern",
]

import logging

import numpy
from astropy.time import Time
from scipy.interpolate import RectBivariateSpline

from rascil.processing_components.calibration.operations import (
    create_gaintable_from_blockvisibility,
)
from rascil.processing_components.util.coordinate_support import hadec_to_azel
from rascil.processing_components.util.geometry import calculate_azel
from rascil.processing_components.visibility.visibility_geometry import (
    calculate_blockvisibility_hourangles,
)

log = logging.getLogger("rascil-logger")


def simulate_gaintable_from_voltage_pattern(
    vis,
    sc,
    vp,
    vis_slices=None,
    order=3,
    elevation_limit=15.0 * numpy.pi / 180.0,
    jones_type="B",
    **kwargs,
):
    """Create gaintables from a list of components and voltage patterns

    :param elevation_limit:
    :param vis_slices:
    :param vis:
    :param sc: Sky components for which pierce points are needed
    :param vp: Voltage pattern in AZELGEO frame, can also be a list of voltage patterns, indexed alphabeticallu
    :param order: order of spline (default is 3)
    :param jones_type: Type of calibration matrix T or G or B
    :return:
    """

    gaintables = [
        create_gaintable_from_blockvisibility(vis, jones_type=jones_type, **kwargs)
        for i in sc
    ]

    nant = gaintables[0].gaintable_acc.nants
    gnchan = gaintables[0].gaintable_acc.nchan
    frequency = gaintables[0].frequency

    # if not isinstance(vp, collections.abc.Iterable):
    if not isinstance(vp, list):
        vp = [vp]

    nchan, npol, ny, nx = vp[0]["pixels"].data.shape
    vnpol = vis.blockvisibility_acc.npol
    vp_types = numpy.unique(vis.configuration.vp_type.data)

    nvp = len(vp_types)

    # Each antennas can have one of a (small) number of different voltage patterns thus
    # allowing for e.g. SKA and MeerKAT antennas for MID
    vp_for_ant = numpy.zeros([nant], dtype=int)
    for ivp in range(nvp):
        for ant in range(nant):
            if vis.configuration.vp_type.data[ant] == vp_types[ivp]:
                vp_for_ant[ant] = ivp

    # We construct interpolators for each voltage pattern type and for each polarisation, and for real, imaginary parts
    if len(vp) == 1:
        vp_types = [0]
    else:
        assert len(vp) == len(vp_types)

    # These are substantial overheads so it is best to call these function as few times as possible.
    # Note that we set up splines only in x, y so there is no interpolation in frequency. In the lookup
    # below, we just use the nearest frequency channel.
    real_spline = [
        [
            [
                RectBivariateSpline(
                    range(ny),
                    range(nx),
                    vp[ivp]["pixels"].data[chan, pol, ...].real,
                    kx=order,
                    ky=order,
                )
                for ivp, _ in enumerate(vp_types)
            ]
            for chan in range(nchan)
        ]
        for pol in range(npol)
    ]
    imag_spline = [
        [
            [
                RectBivariateSpline(
                    range(ny),
                    range(nx),
                    vp[ivp]["pixels"].data[chan, pol, ...].imag,
                    kx=order,
                    ky=order,
                )
                for ivp, _ in enumerate(vp_types)
            ]
            for chan in range(nchan)
        ]
        for pol in range(npol)
    ]

    vp_wcs = [v.image_acc.wcs for v in vp]
    assert vp_wcs[0].wcs.ctype[0] == "AZELGEO long", vp_wcs[0].wcs.ctype[0]
    assert vp_wcs[0].wcs.ctype[1] == "AZELGEO lati", vp_wcs[0].wcs.ctype[1]

    assert vis.configuration.mount[0] == "azel", (
        "Mount %s not supported yet" % vis.configuration.mount[0]
    )

    number_bad = 0
    number_good = 0

    # For each hourangle, we need to calculate the location of a component
    # in AZELGEO. With that we can then look up the relevant gain from the
    # voltage pattern
    gt0 = gaintables[0]
    for row, time in enumerate(gt0.time):
        time_slice = {
            "time": slice(time - gt0.interval[row] / 2, time + gt0.interval[row] / 2)
        }
        v = vis.sel(time_slice)
        utc_time = Time([numpy.average(v.time) / 86400.0], format="mjd", scale="utc")
        azimuth_centre, elevation_centre = calculate_azel(
            v.configuration.location, utc_time, vis.phasecentre
        )
        azimuth_centre = azimuth_centre[0].to("deg").value
        elevation_centre = elevation_centre[0].to("deg").value

        if elevation_centre >= elevation_limit:

            for icomp, comp in enumerate(sc):
                antvp = numpy.zeros([nvp, gnchan, vnpol], dtype="complex")
                antgain = numpy.zeros([nant, gnchan, vnpol], dtype="complex")
                antwt = numpy.zeros([nant, gnchan, vnpol])

                # Calculate the azel of this component
                azimuth_comp, elevation_comp = calculate_azel(
                    v.configuration.location, utc_time, comp.direction
                )
                cosel = numpy.cos(elevation_comp[0]).value
                azimuth_comp = azimuth_comp[0].to("deg").value
                elevation_comp = elevation_comp[0].to("deg").value
                if azimuth_comp - azimuth_centre > 180.0:
                    azimuth_centre += 360.0
                elif azimuth_comp - azimuth_centre < -180.0:
                    azimuth_centre -= 360.0

                for ivp, _ in enumerate(vp_types):
                    for gchan in range(gnchan):
                        try:
                            worldloc = [
                                [
                                    (azimuth_comp - azimuth_centre) * cosel,
                                    elevation_comp - elevation_centre,
                                    vp_wcs[ivp].wcs.crval[2],
                                    frequency.data[gchan],
                                ]
                            ]
                            pixloc = vp_wcs[ivp].wcs_world2pix(worldloc, 0)[0]
                            assert pixloc[0] > 2, f"Error in x pixel : {pixloc}"
                            assert pixloc[0] < nx - 3, f"Error in x pixel : {pixloc}"
                            assert pixloc[1] > 2, f"Error in y pixel : {pixloc}"
                            assert pixloc[1] < ny - 3, f"Error in y pixel : {pixloc}"
                            assert (
                                pixloc[3] > -1
                            ), f"Error in Frequency pixel : {pixloc}"
                            assert (
                                pixloc[3] < nchan
                            ), f"Error in Frequency pixel : {pixloc}"
                            chan = int(round(pixloc[3]))
                            if vnpol > 1:
                                gain = numpy.zeros([npol], dtype="complex")
                                for pol in range(vnpol):
                                    gain[pol] = real_spline[pol][chan][ivp].ev(
                                        pixloc[1], pixloc[0]
                                    ) + 1j * imag_spline[pol][chan][ivp].ev(
                                        pixloc[1], pixloc[0]
                                    )
                                ag = gain.reshape([2, 2])
                                ag = numpy.linalg.inv(ag)
                                antvp[ivp, gchan, :] = ag.reshape([4])
                                number_good += 1
                            else:
                                gain = 0.5 * (
                                    real_spline[0][chan][ivp].ev(pixloc[1], pixloc[0])
                                    + 1j
                                    * imag_spline[0][chan][ivp].ev(pixloc[1], pixloc[0])
                                    + real_spline[3][chan][ivp].ev(pixloc[1], pixloc[0])
                                    + 1j
                                    * imag_spline[3][chan][ivp].ev(pixloc[1], pixloc[0])
                                )
                                if numpy.abs(gain) > 0.0:
                                    antvp[ivp, gchan, 0] = 1.0 / gain
                                    number_good += 1
                                else:
                                    antvp[ivp, gchan, 0] = 0.0
                                    number_bad += 1
                            for ant in range(nant):
                                antgain[ant, ...] = antvp[vp_for_ant[ant], ...]
                            antwt[...] = 1.0
                        except (IndexError, ValueError, AssertionError) as err:
                            print(err)
                            number_bad += 1
                            antgain[...] = 0.0
                            antwt[...] = 0.0

                if vnpol > 1:
                    gaintables[icomp].gain.data[row, :, :, :] = antgain.reshape(
                        [nant, gnchan, 2, 2]
                    )
                    gaintables[icomp].weight.data[row, :, :, :] = antwt.reshape(
                        [nant, gnchan, 2, 2]
                    )
                else:
                    gaintables[icomp].gain.data[row, :, :, :] = antgain.reshape(
                        [nant, gnchan, 1, 1]
                    )
                    gaintables[icomp].weight.data[row, :, :, :] = antwt.reshape(
                        [nant, gnchan, 1, 1]
                    )
                gaintables[icomp].attrs["phasecentre"] = comp.direction
        else:
            for icomp, comp in enumerate(sc):
                gaintables[icomp].gain.data[...] = 1.0 + 0.0j
                gaintables[icomp].weight.data[row, :, :, :] = 0.0
                gaintables[icomp].attrs["phasecentre"] = comp.direction
                number_bad += nant

    assert (
        number_good > 0
    ), "simulate_gaintable_from_voltage_pattern: No points inside the voltage pattern image"
    if number_bad > 0:
        log.warning(
            "simulate_gaintable_from_voltage_pattern: %d points are inside the voltage pattern image"
            % (number_good)
        )
        log.warning(
            "simulate_gaintable_from_voltage_pattern: %d points are outside the voltage pattern image"
            % (number_bad)
        )

    return gaintables


def simulate_gaintable_from_zernikes(
    vis,
    sc,
    vp_list,
    vp_coeffs,
    vis_slices=None,
    order=3,
    elevation_limit=15.0 * numpy.pi / 180.0,
    jones_type="B",
    **kwargs,
):
    """Create gaintables for a set of zernikes

    :param vis:
    :param sc: Sky components for which pierce points are needed
    :param vp: List of Voltage patterns in AZELGEO frame
    :param vp_coeffs: Fractional contribution [nants, nvp]
    :param order: order of spline (default is 3)
    :param jones_type: Type of calibration matrix T or G or B
    :return:
    """

    ntimes = vis.vis.shape[0]
    vp_coeffs = numpy.array(vp_coeffs)
    gaintables = [
        create_gaintable_from_blockvisibility(vis, jones_type=jones_type, **kwargs)
        for i in sc
    ]
    nant = gaintables[0].gaintable_acc.nants

    # assert isinstance(vis, BlockVisibility)
    assert vis.configuration.mount[0] == "azel", (
        "Mount %s not supported yet" % vis.configuration.mount[0]
    )

    # The time in the BlockVisibility is UTC in seconds
    number_bad = 0
    number_good = 0

    # Cache the splines, one per voltage pattern
    real_splines = list()
    imag_splines = list()
    for ivp, vp in enumerate(vp_list):
        assert (
            vp.image_acc.wcs.wcs.ctype[0] == "AZELGEO long"
        ), vp.image_acc.wcs.wcs.ctype[0]
        assert (
            vp.image_acc.wcs.wcs.ctype[1] == "AZELGEO lati"
        ), vp.image_acc.wcs.wcs.ctype[1]

        nchan, npol, ny, nx = vp["pixels"].data.shape
        real_splines.append(
            RectBivariateSpline(
                range(ny),
                range(nx),
                vp["pixels"].data[0, 0, ...].real,
                kx=order,
                ky=order,
            )
        )
        imag_splines.append(
            RectBivariateSpline(
                range(ny),
                range(nx),
                vp["pixels"].data[0, 0, ...].imag,
                kx=order,
                ky=order,
            )
        )

    latitude = vis.configuration.location.lat.rad

    r2d = 180.0 / numpy.pi
    s2r = numpy.pi / 43200.0
    # For each hourangle, we need to calculate the location of a component
    # in AZELGEO. With that we can then look up the relevant gain from the
    # voltage pattern
    for icomp, comp in enumerate(sc):
        gt = gaintables[icomp]
        for row, time in enumerate(gt.time):
            time_slice = {
                "time": slice(time - gt.interval[row] / 2, time + gt.interval[row] / 2)
            }
            vis_sel = vis.sel(time_slice)
            ha = numpy.average(
                calculate_blockvisibility_hourangles(vis_sel).to("rad").value
            )

            # Calculate the az el for this hourangle and the phasecentre declination
            utc_time = Time(
                [numpy.average(vis_sel.time) / 86400.0], format="mjd", scale="utc"
            )
            azimuth_centre, elevation_centre = calculate_azel(
                vis_sel.configuration.location, utc_time, vis.phasecentre
            )
            azimuth_centre = azimuth_centre[0].to("deg").value
            elevation_centre = elevation_centre[0].to("deg").value

            if elevation_centre >= elevation_limit:

                antgain = numpy.zeros([nant], dtype="complex")
                # Calculate the location of the component in AZELGEO, then add the pointing offset
                # for each antenna
                hacomp = comp.direction.ra.rad - vis.phasecentre.ra.rad + ha
                deccomp = comp.direction.dec.rad
                azimuth_comp, elevation_comp = hadec_to_azel(hacomp, deccomp, latitude)

                for ant in range(nant):
                    for ivp, vp in enumerate(vp_list):
                        nchan, npol, ny, nx = vp["pixels"].data.shape
                        wcs_azel = vp.image_acc.wcs.deepcopy()

                        # We use WCS sensible coordinate handling by labelling the axes misleadingly
                        wcs_azel.wcs.crval[0] = azimuth_centre
                        wcs_azel.wcs.crval[1] = elevation_centre
                        wcs_azel.wcs.ctype[0] = "RA---SIN"
                        wcs_azel.wcs.ctype[1] = "DEC--SIN"

                        worldloc = [
                            azimuth_comp * r2d,
                            elevation_comp * r2d,
                            vp.image_acc.wcs.wcs.crval[2],
                            vp.image_acc.wcs.wcs.crval[3],
                        ]
                        try:
                            pixloc = wcs_azel.sub(2).wcs_world2pix([worldloc[:2]], 1)[0]
                            assert pixloc[0] > 2
                            assert pixloc[0] < nx - 3
                            assert pixloc[1] > 2
                            assert pixloc[1] < ny - 3
                            gain = real_splines[ivp].ev(
                                pixloc[1], pixloc[0]
                            ) + 1j * imag_splines[ivp](pixloc[1], pixloc[0])
                            antgain[ant] += vp_coeffs[ant, ivp] * gain
                            number_good += 1
                        except (ValueError, AssertionError):
                            number_bad += 1
                            antgain[ant] = 1.0

                    antgain[ant] = 1.0 / antgain[ant]

                gaintables[icomp].gain.data[row, :, :, :] = antgain[
                    :, numpy.newaxis, numpy.newaxis, numpy.newaxis
                ]
                gaintables[icomp].attrs["phasecentre"] = comp.direction
        else:
            gaintables[icomp].gain.data[...] = 1.0 + 0.0j
            gaintables[icomp].attrs["phasecentre"] = comp.direction
            number_bad += nant

    if number_bad > 0:
        log.warning(
            "simulate_gaintable_from_zernikes: %d points are inside the voltage pattern image"
            % (number_good)
        )
        log.warning(
            "simulate_gaintable_from_zernikes: %d points are outside the voltage pattern image"
            % (number_bad)
        )

    return gaintables
