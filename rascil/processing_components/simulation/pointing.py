""" Functions for simulating pointing errors



"""

__all__ = [
    "simulate_gaintable_from_pointingtable",
    "simulate_pointingtable_from_timeseries",
    "simulate_pointingtable",
]

import logging

import numpy
from astropy.time import Time
from scipy.interpolate import RectBivariateSpline

from rascil.data_models.memory_data_models import PointingTable
from rascil.data_models.parameters import rascil_data_path
from rascil.processing_components.calibration.operations import (
    create_gaintable_from_blockvisibility,
)
from rascil.processing_components.util.geometry import calculate_azel

log = logging.getLogger("rascil-logger")


def simulate_gaintable_from_pointingtable(
    vis,
    sc,
    pt,
    vp,
    vis_slices=None,
    scale=1.0,
    order=3,
    elevation_limit=15.0 * numpy.pi / 180.0,
    **kwargs,
):
    """Create gaintables from a pointing table

    Note that the column "nominal" is not used

    :param vis:
    :param sc: Sky components for which pierce points are needed
    :param pt: Pointing table
    :param vp: Voltage pattern in AZELGEO frame
    :param scale: Multiply the screen by this factor
    :param order: order of spline (default is 3)
    :return:
    """

    nant = vis.blockvisibility_acc.nants
    gaintables = [create_gaintable_from_blockvisibility(vis, **kwargs) for i in sc]

    gnchan = gaintables[0].gaintable_acc.nchan
    frequency = gaintables[0].frequency

    nchan, npol, ny, nx = vp["pixels"].data.shape
    real_spline = [
        [
            RectBivariateSpline(
                range(ny),
                range(nx),
                vp["pixels"].data[chan, pol, ...].real,
                kx=order,
                ky=order,
            )
            for chan in range(nchan)
        ]
        for pol in range(npol)
    ]
    imag_spline = [
        [
            RectBivariateSpline(
                range(ny),
                range(nx),
                vp["pixels"].data[chan, pol, ...].imag,
                kx=order,
                ky=order,
            )
            for chan in range(nchan)
        ]
        for pol in range(npol)
    ]

    assert (
        npol == vis.blockvisibility_acc.npol
    ), "Voltage pattern and visibility have incompatible polarisations"

    # assert isinstance(vis, BlockVisibility)
    assert vp.image_acc.wcs.wcs.ctype[0] == "AZELGEO long", vp.image_acc.wcs.wcs.ctype[
        0
    ]
    assert vp.image_acc.wcs.wcs.ctype[1] == "AZELGEO lati", vp.image_acc.wcs.wcs.ctype[
        1
    ]

    assert vis.configuration.mount[0] == "azel", (
        "Mount %s not supported yet" % vis.configuration.mount[0]
    )

    # The time in the Visibility is UTC in seconds
    number_bad = 0
    number_good = 0
    number_singular = 0

    r2d = 180.0 / numpy.pi
    # For each hourangle, we need to calculate the location of a component
    # in AZELGEO. With that we can then look up the relevant gain from the
    # voltage pattern

    location = vis.attrs["configuration"].attrs["location"]
    for row, time in enumerate(pt["time"]):
        time_slice = {
            "time": slice(time - pt["interval"][row] / 2, time + pt.interval[row] / 2)
        }
        pt_sel = pt.sel(time_slice)
        if len(pt_sel["time"]) > 0:
            pointing_ha = pt_sel["pointing"].data[0, ...]
            utc_time = Time(
                [numpy.average(pt_sel["time"]) / 86400.0], format="mjd", scale="utc"
            )
            azimuth_centre, elevation_centre = calculate_azel(
                location, utc_time, vis.phasecentre
            )
            azimuth_centre = azimuth_centre[0].to("rad").value
            elevation_centre = elevation_centre[0].to("rad").value

            # Calculate the az el for this hourangle and the phasecentre declination
            for icomp, comp in enumerate(sc):
                gt_sel = gaintables[icomp].sel(time_slice)
                nrec = gt_sel.gaintable_acc.nrec
                if elevation_centre >= elevation_limit:

                    antgain = numpy.zeros([nant, gnchan, npol], dtype="complex")

                    # Calculate the azel of this component
                    azimuth_comp, elevation_comp = calculate_azel(
                        location, utc_time, comp.direction
                    )
                    azimuth_comp = azimuth_comp[0].to("rad").value
                    elevation_comp = elevation_comp[0].to("rad").value
                    # We now add the pointing to the calculated az, el of the component
                    for ant in range(nant):
                        wcs_azel = vp.image_acc.wcs.deepcopy()
                        az_comp = azimuth_comp + pointing_ha[ant, 0, 0, 0] / numpy.cos(
                            elevation_centre
                        )
                        el_comp = elevation_comp + pointing_ha[ant, 0, 0, 1]

                        if az_comp - azimuth_comp > numpy.pi:
                            azimuth_comp += 2.0 * numpy.pi
                        if az_comp - azimuth_comp < -numpy.pi:
                            azimuth_comp -= 2.0 * numpy.pi

                        # We use WCS sensible coordinate handling by labelling the axes misleadingly
                        wcs_azel.wcs.crval[0] = az_comp * r2d
                        wcs_azel.wcs.crval[1] = el_comp * r2d
                        wcs_azel.wcs.ctype[0] = "RA---SIN"
                        wcs_azel.wcs.ctype[1] = "DEC--SIN"

                        try:
                            # Unwrap azimuths
                            if azimuth_comp - az_comp > numpy.pi:
                                azimuth_comp -= 2.0 * numpy.pi
                            elif azimuth_comp - az_comp <= -numpy.pi:
                                azimuth_comp += 2.0 * numpy.pi
                            if azimuth_comp > numpy.pi:
                                azimuth_comp -= 2.0 * numpy.pi
                            elif azimuth_comp <= -numpy.pi:
                                azimuth_comp += 2.0 * numpy.pi

                            for gchan in range(gnchan):
                                gain = numpy.zeros([npol], dtype="complex")
                                worldloc = [
                                    azimuth_comp * r2d,
                                    elevation_comp * r2d,
                                    vp.image_acc.wcs.wcs.crval[2],
                                    frequency[gchan],
                                ]
                                pixloc = wcs_azel.wcs_world2pix([worldloc], 0)[0]
                                assert pixloc[0] > 2
                                assert pixloc[0] < nx - 3
                                assert pixloc[1] > 2
                                assert pixloc[1] < ny - 3
                                chan = int(round(pixloc[3]))
                                if nchan == 1:
                                    chan = 0
                                for pol in range(npol):
                                    gain[pol] = real_spline[pol][chan].ev(
                                        pixloc[1], pixloc[0]
                                    ) + 1j * imag_spline[pol][chan].ev(
                                        pixloc[1], pixloc[0]
                                    )
                                if nrec == 2:
                                    ag = gain.reshape([2, 2])
                                    ag = numpy.linalg.inv(ag)
                                    antgain[ant, gchan, :] = ag.reshape([4])
                                elif nrec == 1:
                                    antgain[ant, gchan, 0] = 1.0 / gain
                                else:
                                    raise ValueError(
                                        "Illegal number of receptors: {}".format(nrec)
                                    )
                                number_good += 1
                        except (ValueError, AssertionError, IndexError):
                            number_bad += 1
                            antgain[ant, :, :] = 0.0
                        except (numpy.linalg.LinAlgError):
                            number_singular += 1
                            antgain[ant, :, :] = 0.0

                        gt_sel["gain"].data[:, :, :, :] = antgain[:, :, :].reshape(
                            [nant, gnchan, nrec, nrec]
                        )
                        gt_sel.attrs["phasecentre"] = comp.direction
                else:
                    gt_sel["gain"].data[...] = 1.0 + 0.0j
                    gt_sel.attrs["phasecentre"] = comp.direction
                    number_bad += nant
        else:
            log.warning("Zero length pointing interval")

    if number_bad > 0:
        log.debug(
            "simulate_gaintable_from_pointingtable: %d points inside the voltage pattern image"
            % (number_good)
        )
        log.debug(
            "simulate_gaintable_from_pointingtable: %d points outside the voltage pattern image will be ignored"
            % (number_bad)
        )
    if number_singular > 0:
        log.debug(
            "simulate_gaintable_from_pointingtable: %d points have singular gain"
            % (number_singular)
        )

    return gaintables


def simulate_pointingtable(
    pt: PointingTable,
    pointing_error,
    static_pointing_error=None,
    global_pointing_error=None,
    seed=None,
    **kwargs,
) -> PointingTable:
    """Simulate a gain table

    :type pt: PointingTable
    :param pointing_error: std of normal distribution (radians)
    :param static_pointing_error: std of normal distribution (radians)
    :param global_pointing_error: 2-vector of global pointing error (rad)
    :param kwargs:
    :return: PointingTable

    """

    from numpy.random import default_rng

    if seed is None:
        rng = default_rng(1805550721)
    else:
        rng = default_rng(seed)

    if static_pointing_error is None:
        static_pointing_error = [0.0, 0.0]

    r2s = 180.0 * 3600.0 / numpy.pi
    pt["pointing"].data = numpy.zeros(pt["pointing"].data.shape)

    ntimes, nant, nchan, nrec, _ = pt["pointing"].data.shape
    if pointing_error > 0.0:
        log.debug(
            "simulate_pointingtable: Simulating dynamic pointing error = %g (rad) %g (arcsec)"
            % (pointing_error, r2s * pointing_error)
        )

        pt["pointing"].data += rng.normal(
            0.0, pointing_error, pt["pointing"].data.shape
        )
    if (abs(static_pointing_error[0]) > 0.0) or (abs(static_pointing_error[1]) > 0.0):
        log.debug(
            "simulate_pointingtable: Simulating static pointing error = (%g, %g) (rad) (%g, %g)(arcsec)"
            % (
                static_pointing_error[0],
                static_pointing_error[1],
                r2s * static_pointing_error[0],
                r2s * static_pointing_error[1],
            )
        )

        static_pe = numpy.zeros(pt["pointing"].data.shape[1:])
        static_pe[..., 0] = rng.normal(
            0.0, static_pointing_error[0], static_pe[..., 0].shape
        )[numpy.newaxis, ...]
        static_pe[..., 1] = rng.normal(
            0.0, static_pointing_error[1], static_pe[..., 1].shape
        )[numpy.newaxis, ...]
        pt["pointing"].data += static_pe

    if global_pointing_error is not None:

        log.debug(
            "simulate_pointingtable: Simulating global pointing error = [%g, %g] (rad) [%g,s %g] (arcsec)"
            % (
                global_pointing_error[0],
                global_pointing_error[1],
                r2s * global_pointing_error[0],
                r2s * global_pointing_error[1],
            )
        )
        pt["pointing"].data[..., :] += global_pointing_error

    return pt


def simulate_pointingtable_from_timeseries(
    pt,
    type="wind",
    time_series_type="precision",
    pointing_directory=None,
    reference_pointing=False,
    seed=None,
):
    """Create a pointing table with time series created from PSD.

    :param pt: Pointing table to be filled
    :param type: Type of pointing: 'tracking' or 'wind'
    :param time_series_type: Type of wind condition precision|standard|degraded
    :param pointing_directory: Name of pointing file directory
    :param reference_pointing: Use reference pointing?
    :return:
    """
    from numpy.random import default_rng

    if seed is None:
        rng = default_rng(1805550721)
    else:
        rng = default_rng(seed)

    if pointing_directory is None:
        pointing_directory = rascil_data_path("models/%s" % time_series_type)
    else:
        pointing_directory = pointing_directory + "/%s" % (time_series_type)

    pt["pointing"].data = numpy.zeros(pt["pointing"].data.shape)

    ntimes, nant, nchan, nrec, _ = pt["pointing"].data.shape

    # Use az and el at the beginning of this pointingtable
    axis_values = pt.nominal[0, 0, 0, 0, 0]
    el = pt.nominal[0, 0, 0, 0, 1]

    el_deg = el * 180.0 / numpy.pi
    az_deg = axis_values * 180.0 / numpy.pi

    if el_deg < 30.0:
        el_deg = 15.0
    elif el_deg < (90.0 + 45.0) / 2.0:
        el_deg = 45.0
    else:
        el_deg = 90.0

    if abs(az_deg) < 45.0 / 2.0:
        az_deg = 0.0
    elif abs(az_deg) < (45.0 + 90.0) / 2.0:
        az_deg = 45.0
    elif abs(az_deg) < (90.0 + 135.0) / 2.0:
        az_deg = 90.0
    elif abs(az_deg) < (135.0 + 180.0) / 2.0:
        az_deg = 135.0
    else:
        az_deg = 180.0

    pointing_file = "%s/El%dAz%d.dat" % (pointing_directory, int(el_deg), int(az_deg))
    log.info(
        "simulate_pointingtable_from_timeseries: Reading wind PSD from %s"
        % pointing_file
    )
    try:
        psd = numpy.loadtxt(pointing_file)
    except OSError:
        raise ValueError("Pointing file %s not found." % pointing_file)

    # define some arrays
    freq = psd[:, 0]
    axesdict = {"az": psd[:, 1], "el": psd[:, 2], "pxel": psd[:, 3], "pel": psd[:, 4]}

    if type == "tracking":
        axes = ["az", "el"]
    elif type == "wind":
        axes = ["pxel", "pel"]
    else:
        raise ValueError("Pointing type %s not known." % type)

    freq_interval = 0.0001

    for axis in axes:

        axis_values = axesdict[axis]

        if (axis == "az") or (axis == "el"):
            # determine index of maximum PSD value; add 50 for better fit
            axis_values_max_index = (
                numpy.argwhere(axis_values == numpy.max(axis_values))[0][0] + 50
            )
            axis_values_max_index = min(axis_values_max_index, len(axis_values))
            # max_freq = 2.0 / pt.interval[0]
            max_freq = 0.4
            freq_max_index = numpy.argwhere(freq > max_freq)[0][0]
        else:
            break_freq = 0.01  # not max; just a break
            axis_values_max_index = numpy.argwhere(freq > break_freq)[0][0]
            # max_freq = 2.0 / pt.interval[0]
            max_freq = 0.1
            freq_max_index = numpy.argwhere(freq > max_freq)[0][0]

        # construct regularly-spaced frequencies
        regular_freq = numpy.arange(freq[0], freq[freq_max_index], freq_interval)

        regular_axis_values_max_index = numpy.argwhere(
            numpy.abs(regular_freq - freq[axis_values_max_index])
            == numpy.min(numpy.abs(regular_freq - freq[axis_values_max_index]))
        )[0][0]

        # print ('Frequency break: ', freq[az_max_index])
        # print ('Max frequency: ', max_freq)
        #
        # print ('New frequency break: ', regular_freq[regular_az_max_index])
        # print ('New max frequency: ', regular_freq[-1])

        if axis_values_max_index >= freq_max_index:
            raise ValueError(
                "Frequency break is higher than highest frequency; select a lower break."
            )

        # use original frequency break and max frequency to fit function
        # fit polynomial to psd up to max value
        import warnings
        from numpy import RankWarning

        warnings.simplefilter("ignore", RankWarning)

        p_axis_values1 = numpy.polyfit(
            freq[:axis_values_max_index],
            numpy.log(axis_values[:axis_values_max_index]),
            5,
        )
        f_axis_values1 = numpy.poly1d(p_axis_values1)
        # fit polynomial to psd beyond max value
        p_axis_values2 = numpy.polyfit(
            freq[axis_values_max_index:freq_max_index],
            numpy.log(axis_values[axis_values_max_index:freq_max_index]),
            5,
        )
        f_axis_values2 = numpy.poly1d(p_axis_values2)

        # use new frequency break and max frequency to apply function (ensures equal spacing of frequency intervals)

        # resampled to construct regularly-spaced frequencies
        regular_axis_values1 = numpy.exp(
            f_axis_values1(regular_freq[:regular_axis_values_max_index])
        )
        regular_axis_values2 = numpy.exp(
            f_axis_values2(regular_freq[regular_axis_values_max_index:])
        )

        # join
        regular_axis_values = numpy.append(regular_axis_values1, regular_axis_values2)

        m0 = len(regular_axis_values)

        #  check rms of resampled PSD
        # df = regular_freq[1:]-regular_freq[:-1]
        # psd2rms_pxel = numpy.sqrt(numpy.sum(regular_az[:-1]*df))
        # print ('Calculate rms of resampled PSD: ', psd2rms_pxel)

        original_regular_freq = regular_freq
        original_regular_axis_values = regular_axis_values
        # get amplitudes from psd values

        if (regular_axis_values < 0).any():
            raise ValueError(
                "Resampling returns negative power values; change fit range."
            )

        amp_axis_values = numpy.sqrt(regular_axis_values * 2 * freq_interval)
        # need to scale PSD by 2* frequency interval before square rooting, then by number of modes in resampled PSD

        # Now we generate some random phases
        for ant in range(nant):
            regular_freq = original_regular_freq
            regular_axis_values = original_regular_axis_values
            phi_axis_values = rng.random(size=len(regular_axis_values)) * 2 * numpy.pi
            # create complex array
            z_axis_values = amp_axis_values * numpy.exp(1j * phi_axis_values)  # polar
            # make symmetrical frequencies
            mirror_z_axis_values = numpy.copy(z_axis_values)
            # make complex conjugates
            mirror_z_axis_values.imag -= 2 * z_axis_values.imag
            # make negative frequencies
            mirror_regular_freq = -regular_freq
            # join
            z_axis_values = numpy.append(z_axis_values, mirror_z_axis_values[::-1])
            regular_freq = numpy.append(regular_freq, mirror_regular_freq[::-1])

            # add a 0 Fourier term
            zav = z_axis_values
            z_axis_values = numpy.zeros([len(zav) + 1]).astype("complex")
            z_axis_values[1:] = zav

            # perform inverse fft
            ts = numpy.fft.ifft(z_axis_values)

            # set up and check scalings
            Dt = pt.interval[0]
            ts = numpy.real(ts)
            ts *= m0  # the result is scaled by number of points in the signal, so multiply - real part - by this

            # The output of the iFFT will be a random time series on the finite
            # (bounded, limited) time interval t = 0 to tmax = (N-1) X Dt, #
            # where Dt = 1 / (2 X Fmax)

            # scale to time interval

            # Convert from arcsec to radians
            ts *= numpy.pi / (180.0 * 3600.0)

            # We take reference pointing to mean that the pointing errors are zero at the beginning
            # of the set of integrations
            if reference_pointing:
                ts[:] -= ts[0]

            #            pt.data['time'] = times[:ntimes]
            if axis == "az":
                pt["pointing"].data[:, ant, :, :, 0] = ts[
                    :ntimes, numpy.newaxis, numpy.newaxis, ...
                ]
            elif axis == "el":
                pt["pointing"].data[:, ant, :, :, 1] = ts[
                    :ntimes, numpy.newaxis, numpy.newaxis, ...
                ]
            elif axis == "pxel":
                pt["pointing"].data[:, ant, :, :, 0] = ts[
                    :ntimes, numpy.newaxis, numpy.newaxis, ...
                ]
            elif axis == "pel":
                pt["pointing"].data[:, ant, :, :, 1] = ts[
                    :ntimes, numpy.newaxis, numpy.newaxis, ...
                ]
            else:
                raise ValueError("Unknown axis %s" % axis)

    return pt
