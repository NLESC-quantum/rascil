"""
Functions to create primary beam and voltage pattern models
"""

__all__ = [
    "set_pb_header",
    "create_pb",
    "create_pb_generic",
    "create_vp",
    "create_vp_generic",
    "create_vp_generic_numeric",
    "create_low_test_beam",
    "create_low_test_vp",
    "create_mid_allsky",
    "convert_azelvp_to_radec",
    "normalise_vp",
]

import collections
import logging
import copy

import numpy
import xarray

from astropy.coordinates import SkyCoord
import astropy.units as u

from rascil.data_models import Image, PolarisationFrame
from rascil.data_models.parameters import rascil_data_path
from rascil.processing_components.image.operations import (
    import_image_from_fits,
    reproject_image,
    scale_and_rotate_image,
    create_image,
)
from rascil.processing_components.image.operations import (
    create_image_from_array,
    create_empty_image_like,
    ifft_griddata_to_image,
    fft_image_to_griddata,
    pad_image,
)
from rascil import phyconst

log = logging.getLogger("rascil-logger")

# Copyright of function zernIndex and zernike_noll belongs to its respectful owner
# zernIndex and zernike_noll are the functions of AOTools.zernike
# AOTools: https://github.com/AOtools/aotools
# L-GPL 3.0


def zernIndex(j):
    """
    Find the [n,m] list giving the radial order n and azimuthal order
    of the Zernike polynomial of Noll index j.

    Parameters:
        j (int): The Noll index for Zernike polynomials

    Returns:
        list: n, m values
    """
    n = int((-1.0 + numpy.sqrt(8 * (j - 1) + 1)) / 2.0)
    p = j - (n * (n + 1)) / 2.0
    k = n % 2
    m = int((p + k) / 2.0) * 2 - k

    if m != 0:
        if j % 2 == 0:
            s = 1
        else:
            s = -1
        m *= s

    return [n, m]


def circle(radius, size, circle_centre=(0, 0), origin="middle"):
    """
    Create a 2-D array: elements equal 1 within a circle and 0 outside.

    The default centre of the coordinate system is in the middle of the array:
    circle_centre=(0,0), origin="middle"
    This means:
    if size is odd  : the centre is in the middle of the central pixel
    if size is even : centre is in the corner where the central 4 pixels meet

    origin = "corner" is used e.g. by psfAnalysis:radialAvg()

    Examples: ::

        circle(1,5) circle(0,5) circle(2,5) circle(0,4) circle(0.8,4) circle(2,4)
          00000       00000       00100       0000        0000          0110
          00100       00000       01110       0000        0110          1111
          01110       00100       11111       0000        0110          1111
          00100       00000       01110       0000        0000          0110
          00000       00000       00100

        circle(1,5,(0.5,0.5))   circle(1,4,(0.5,0.5))
           .-->+
           |  00000               0000
           |  00000               0010
          +V  00110               0111
              00110               0010
              00000

    Parameters:
        radius (float)       : radius of the circle
        size (int)           : size of the 2-D array in which the circle lies
        circle_centre (tuple): coords of the centre of the circle
        origin (str)  : where is the origin of the coordinate system
                               in which circle_centre is given;
                               allowed values: {"middle", "corner"}

    Returns:
        ndarray (float64) : the circle array
    """
    # (2) Generate the output array:
    C = numpy.zeros((size, size))

    # (3.a) Generate the 1-D coordinates of the pixel's centres:
    # coords = numpy.linspace(-size/2.,size/2.,size) # Wrong!!:
    # size = 5: coords = array([-2.5 , -1.25,  0.  ,  1.25,  2.5 ])
    # size = 6: coords = array([-3. , -1.8, -0.6,  0.6,  1.8,  3. ])
    # (2015 Mar 30; delete this comment after Dec 2015 at the latest.)

    # Before 2015 Apr 7 (delete 2015 Dec at the latest):
    # coords = numpy.arange(-size/2.+0.5, size/2.-0.4, 1.0)
    # size = 5: coords = array([-2., -1.,  0.,  1.,  2.])
    # size = 6: coords = array([-2.5, -1.5, -0.5,  0.5,  1.5,  2.5])

    coords = numpy.arange(0.5, size, 1.0)
    # size = 5: coords = [ 0.5  1.5  2.5  3.5  4.5]
    # size = 6: coords = [ 0.5  1.5  2.5  3.5  4.5  5.5]

    # (3.b) Just an internal sanity check:
    if len(coords) != size:
        raise Exception(
            "len(coords) = {0}, ".format(len(coords))
            + "size = {0}. They must be equal.".format(size)
            + '\n           Debug the line "coords = ...".'
        )

    # (3.c) Generate the 2-D coordinates of the pixel's centres:
    x, y = numpy.meshgrid(coords, coords)

    # (3.d) Move the circle origin to the middle of the grid, if required:
    if origin == "middle":
        x -= size / 2.0
        y -= size / 2.0

    # (3.e) Move the circle centre to the alternative position, if provided:
    x -= circle_centre[0]
    y -= circle_centre[1]

    # (4) Calculate the output:
    # if distance(pixel's centre, circle_centre) <= radius:
    #     output = 1
    # else:
    #     output = 0
    mask = x * x + y * y <= radius * radius
    C[mask] = 1

    # (5) Return:
    return C


def zernikeRadialFunc(n, m, r):
    """
    Fucntion to calculate the Zernike radial function

    Parameters:
        n (int): Zernike radial order
        m (int): Zernike azimuthal order
        r (ndarray): 2-d array of radii from the centre the array

    Returns:
        ndarray: The Zernike radial function
    """

    R = numpy.zeros(r.shape)
    for i in range(0, int((n - m) / 2) + 1):

        R += numpy.array(
            r ** (n - 2 * i)
            * (((-1) ** (i)) * numpy.math.factorial(n - i))
            / (
                numpy.math.factorial(i)
                * numpy.math.factorial(0.5 * (n + m) - i)
                * numpy.math.factorial(0.5 * (n - m) - i)
            ),
            dtype="float",
        )
    return R


def zernike_nm(n, m, N):
    """
    Creates the Zernike polynomial with radial index, n, and azimuthal index, m.

    Args:
       n (int): The radial order of the zernike mode
       m (int): The azimuthal order of the zernike mode
       N (int): The diameter of the zernike more in pixels
    Returns:
       ndarray: The Zernike mode
    """
    coords = (numpy.arange(N) - N / 2.0 + 0.5) / (N / 2.0)
    X, Y = numpy.meshgrid(coords, coords)
    R = numpy.sqrt(X ** 2 + Y ** 2)
    theta = numpy.arctan2(Y, X)

    if m == 0:
        Z = numpy.sqrt(n + 1) * zernikeRadialFunc(n, 0, R)
    else:
        if m > 0:  # j is even
            Z = (
                numpy.sqrt(2 * (n + 1))
                * zernikeRadialFunc(n, m, R)
                * numpy.cos(m * theta)
            )
        else:  # i is odd
            m = abs(m)
            Z = (
                numpy.sqrt(2 * (n + 1))
                * zernikeRadialFunc(n, m, R)
                * numpy.sin(m * theta)
            )

    # clip
    Z = Z * numpy.less_equal(R, 1.0)

    return Z * circle(N / 2.0, N)


def zernike_noll(j, N):
    """
    Creates the Zernike polynomial with mode index j,
    where j = 1 corresponds to piston.

    Args:
       j (int): The noll j number of the zernike mode
       N (int): The diameter of the zernike more in pixels
    Returns:
       ndarray: The Zernike mode
    """

    n, m = zernIndex(j)
    return zernike_nm(n, m, N)


def set_pb_header(pb, use_local=True):
    """Fill in PB header correctly for local coordinates.

    There is no convention on how to represent primary beams. We use axes 'AZELGEO long' and 'AZELGEO lati'

    :param pb:
    :return:
    """
    if use_local:

        nchan, npol, ny, nx = pb["pixels"].shape
        wcs = pb.image_acc.wcs
        wcs.wcs.ctype[0] = "AZELGEO long"
        wcs.wcs.ctype[1] = "AZELGEO lati"
        wcs.wcs.crval[0] = 0.0
        wcs.wcs.crval[1] = 0.0
        wcs.wcs.crpix[0] = nx // 2
        wcs.wcs.crpix[1] = ny // 2
        pb = create_image_from_array(
            pb["pixels"].data,
            wcs=wcs,
            polarisation_frame=pb.image_acc.polarisation_frame,
        )

    return pb


def gauss(x0, y0, amp, sigma, rho, diff, a):
    """
    2D gaussian

    :param a: Grid of aperture plane coordinates
    """
    dx = a[..., 0] - x0
    dy = a[..., 1] - y0
    r = numpy.hypot(dx, dy)
    return amp * numpy.exp(
        -1.0
        / (2 * sigma ** 2)
        * (r ** 2 + rho * (dx * dy) + diff * (dx ** 2 - dy ** 2))
    )


def ft_disk(r):
    from scipy.special import jn  # pylint: disable=no-name-in-module

    result = numpy.zeros_like(r, dtype="complex")
    result[r > 0] = 2.0 * jn(1, r[r > 0]) / r[r > 0]
    rsmall = 1e-9
    result[r == 0] = 2.0 * jn(1, rsmall) / rsmall
    return result


def tapered_disk(r, radius, blockage=0.0, taper="gaussian", edge=1.0):
    result = numpy.zeros_like(r, dtype="complex")
    if taper == "gaussian":
        # exp(-gscale*radius**2) = taper
        gscale = -numpy.log(edge) / radius ** 2
        result[r < radius] = numpy.exp(-gscale * r[r < radius] ** 2)
    result[r < blockage] = 0.0
    return result


def create_vp(
    model=None,
    telescope="MID",
    pointingcentre=None,
    padding=4,
    use_local=True,
    fixpol=True,
):
    """Create an image containing the dish voltage pattern for a number of cases

    :param model: Template image (Can be None for some cases)
    :param telescope:
    :return: Primary beam image
    """

    if telescope == "MID_GAUSS":
        if model is None:
            raise ValueError(f"Need model image for MID_GAUSS telescope type")

        log.debug(
            "create_vp: Using numeric tapered Gaussian model for MID voltage pattern"
        )

        edge = numpy.power(10, -0.6)
        return create_vp_generic_numeric(
            model,
            pointingcentre=pointingcentre,
            diameter=15.0,
            blockage=0.0,
            edge=edge,
            padding=padding,
            use_local=use_local,
        )
    elif telescope == "MID":
        if model is None:
            raise ValueError(f"Need model image for MID telescope type")

        log.debug("create_vp: Using no taper analytic model for MID voltage pattern")
        return create_vp_generic(
            model,
            pointingcentre=pointingcentre,
            diameter=15.0,
            blockage=0.0,
            use_local=use_local,
        )
    elif telescope == "MID_FEKO_B1LOW" or telescope == "MID_B1LOW":
        log.debug("create_vp: Using FEKO model for MID voltage pattern")
        real_vp = import_image_from_fits(
            rascil_data_path("models/MID_FEKO_VP_B1_45_0365_real.fits"), fixpol=fixpol
        )
        imag_vp = import_image_from_fits(
            rascil_data_path("models/MID_FEKO_VP_B1_45_0365_imag.fits"), fixpol=fixpol
        )
        real_vp["pixels"].data = real_vp["pixels"].data + 1j * imag_vp["pixels"].data
        real_vp["pixels"].data /= numpy.max(numpy.abs(real_vp["pixels"].data))
        return real_vp
    elif telescope == "MID_FEKO_B1" or telescope == "MID_B1":
        log.debug("create_vp: Using FEKO model for MID voltage pattern")
        real_vp = import_image_from_fits(
            rascil_data_path("models/MID_FEKO_VP_B1_45_0765_real.fits"), fixpol=fixpol
        )
        imag_vp = import_image_from_fits(
            rascil_data_path("models/MID_FEKO_VP_B1_45_0765_imag.fits"), fixpol=fixpol
        )
        real_vp["pixels"].data = real_vp["pixels"].data + 1j * imag_vp["pixels"].data
        real_vp["pixels"].data /= numpy.max(numpy.abs(real_vp["pixels"].data))
        return real_vp
    elif telescope == "MID_FEKO_B2" or telescope == "MID_B2":
        log.debug("create_vp: Using FEKO model for MID voltage pattern")
        real_vp = import_image_from_fits(
            rascil_data_path("models/MID_FEKO_VP_B2_45_1360_real.fits"), fixpol=fixpol
        )
        imag_vp = import_image_from_fits(
            rascil_data_path("models/MID_FEKO_VP_B2_45_1360_imag.fits"), fixpol=fixpol
        )
        real_vp["pixels"].data = real_vp["pixels"].data + 1j * imag_vp["pixels"].data
        real_vp["pixels"].data /= numpy.max(numpy.abs(real_vp["pixels"].data))
        return real_vp
    elif telescope == "MID_FEKO_Ku" or telescope == "MID_Ku":
        log.debug("create_vp: Using FEKO model for MID voltage pattern")
        real_vp = import_image_from_fits(
            rascil_data_path("models/MID_FEKO_VP_Ku_45_12179_real.fits"), fixpol=fixpol
        )
        imag_vp = import_image_from_fits(
            rascil_data_path("models/MID_FEKO_VP_Ku_45_12179_imag.fits"), fixpol=fixpol
        )
        real_vp["pixels"].data = real_vp["pixels"].data + 1j * imag_vp["pixels"].data
        real_vp["pixels"].data /= numpy.max(numpy.abs(real_vp["pixels"].data))
        return real_vp
    elif telescope == "MEERKAT_B2":
        log.debug("create_vp: Using MEERKAT voltage pattern")
        real_vp = import_image_from_fits(
            rascil_data_path("models/MeerKAT_VP_60_1360_real.fits"), fixpol=fixpol
        )
        imag_vp = import_image_from_fits(
            rascil_data_path("models/MeerKAT_VP_60_1360_imag.fits"), fixpol=fixpol
        )
        real_vp["pixels"].data = real_vp["pixels"].data + 1j * imag_vp["pixels"].data
        real_vp["pixels"].data /= numpy.max(numpy.abs(real_vp["pixels"].data))
        return real_vp
    elif telescope == "MEERKAT_B1":
        log.debug("create_vp: Using MID FEKO model for MEERKAT B1 voltage pattern")
        real_vp = import_image_from_fits(
            rascil_data_path("models/MID_FEKO_VP_B1_45_0765_real.fits"), fixpol=fixpol
        )
        imag_vp = import_image_from_fits(
            rascil_data_path("models/MID_FEKO_VP_B1_45_0765_imag.fits"), fixpol=fixpol
        )
        real_vp["pixels"].data = real_vp["pixels"].data + 1j * imag_vp["pixels"].data
        real_vp["pixels"].data /= numpy.max(numpy.abs(real_vp["pixels"].data))
        return real_vp
    elif telescope[0:3] == "LOW":
        return create_low_test_vp(model)
    elif telescope[0:3] == "VLA":
        return create_vp_generic(
            model,
            pointingcentre=pointingcentre,
            diameter=25.0,
            blockage=1.8,
            use_local=use_local,
        )
    elif telescope[0:5] == "ASKAP":
        return create_vp_generic(
            model,
            pointingcentre=pointingcentre,
            diameter=12.0,
            blockage=1.0,
            use_local=use_local,
        )
    else:
        raise NotImplementedError(
            "Telescope %s has no voltage pattern model" % telescope
        )


def create_pb(model, telescope="MID", pointingcentre=None, use_local=True):
    """Create an image containing the primary beam for a number of cases

    :param model: Template image
    :param telescope: 'VLA' or 'ASKAP'
    :return: Primary beam image
    """
    beam = create_vp(model, telescope, pointingcentre, use_local=use_local)
    beam["pixels"].data = numpy.real(
        beam["pixels"].values * numpy.conjugate(beam["pixels"].values)
    )

    set_pb_header(beam, use_local=use_local)
    return beam


def mosaic_pb(model, telescope, pointingcentres, use_local=True):
    """Create a mosaic effective primary beam by adding primary beams for a set of pointing centres

    Note that the addition is root sum of squares

    :param model:  Template image
    :param telescope:
    :param pointingcentres: list of pointing centres
    :return:
    """
    # assert isinstance(pointingcentres, collections.abc.Iterable), "Need a list of pointing centres"
    sumpb = create_empty_image_like(model)
    for pc in pointingcentres:
        pb = create_pb(model, telescope, pointingcentre=pc, use_local=use_local)
        sumpb["pixels"].data += pb["pixels"].data ** 2
    sumpb["pixels"].data = numpy.sqrt(sumpb["pixels"].data)
    return sumpb


def create_pb_generic(
    model, pointingcentre=None, diameter=25.0, blockage=1.8, use_local=True
):
    """Create a generic analytical model of the primary beam

    Feeed legs are ignored

    :param model:
    :param diameter: Diameter of dish (m)
    :param blockage: Diameter of blockage
    :return:
    """
    beam = create_vp_generic(
        model, pointingcentre, diameter, blockage, use_local=use_local
    )
    beam["pixels"].data = numpy.real(
        beam["pixels"].data * numpy.conjugate(beam["pixels"].data)
    )
    set_pb_header(beam, use_local=use_local)
    return beam


def create_vp_generic(
    model, pointingcentre=None, diameter=25.0, blockage=1.8, use_local=True
):
    """Create a generic analytical model of the voltage pattern

    Feeed legs are ignored

    :param model:
    :param diameter: Diameter of dish (m)
    :param blockage: Diameter of blockage
    :return:
    """

    beam = create_empty_image_like(model)
    beam["pixels"].data = numpy.zeros(beam["pixels"].data.shape, dtype="complex")

    nchan, npol, ny, nx = model["pixels"].shape

    if pointingcentre is not None:
        cx, cy = pointingcentre.to_pixel(model.image_acc.wcs, origin=0)
    else:
        cx, cy = (
            beam.image_acc.wcs.sub(2).wcs.crpix[0] - 1,
            beam.image_acc.wcs.sub(2).wcs.crpix[1] - 1,
        )

    for chan in range(nchan):

        # The frequency axis is the second to last in the beam
        frequency = model.image_acc.wcs.sub(["spectral"]).wcs_pix2world([chan], 0)[0]
        wavelength = phyconst.c_m_s / frequency

        d2r = numpy.pi / 180.0
        scale = d2r * numpy.abs(beam.image_acc.wcs.sub(2).wcs.cdelt[0])
        xx, yy = numpy.meshgrid(
            scale * (numpy.arange(nx) - cx), scale * (numpy.arange(ny) - cy)
        )
        # Radius of each cell in radians
        rr = numpy.sqrt(xx ** 2 + yy ** 2)

        blockage_factor = (blockage / diameter) ** 2

        if beam.image_acc.polarisation_frame == PolarisationFrame("linear"):
            pols = [0, 3]
        elif beam.image_acc.polarisation_frame == PolarisationFrame("circular"):
            pols = [0, 3]
        elif beam.image_acc.polarisation_frame == PolarisationFrame("linearnp"):
            pols = [0, 1]
        elif beam.image_acc.polarisation_frame == PolarisationFrame("circularnp"):
            pols = [0, 1]
        elif beam.image_acc.polarisation_frame == PolarisationFrame("stokesI"):
            pols = [0]
        elif beam.image_acc.polarisation_frame == PolarisationFrame("stokesIQUV"):
            pols = [0, 1, 2, 3]
        elif beam.image_acc.polarisation_frame == PolarisationFrame("stokesIQ"):
            pols = [0, 1]
        elif beam.image_acc.polarisation_frame == PolarisationFrame("stokesIV"):
            pols = [0, 1]
        else:
            raise ValueError(
                "Polarisation frame {}, cannot set all voltage pattern polarisations".format(
                    beam.image_acc.polarisation_frame
                )
            )

        reflector = ft_disk(rr * numpy.pi * diameter / wavelength)
        blockage = ft_disk(rr * numpy.pi * blockage / wavelength)
        combined = reflector - blockage_factor * blockage

        for pol in pols:
            beam["pixels"].data[chan, pol, ...] = combined

    beam = set_pb_header(beam, use_local=use_local)

    if use_local:
        assert (
            beam.image_acc.wcs.wcs.ctype[0] == "AZELGEO long"
        ), beam.image_acc.wcs.wcs.ctype[0]
        assert (
            beam.image_acc.wcs.wcs.ctype[1] == "AZELGEO lati"
        ), beam.image_acc.wcs.wcs.ctype[1]

    return beam


def create_vp_generic_numeric(
    model,
    pointingcentre=None,
    diameter=15.0,
    blockage=0.0,
    taper="gaussian",
    edge=0.03162278,
    zernikes=None,
    padding=4,
    use_local=True,
):
    """
    Make an image like model and fill it with an analytical model of the primary beam

    The elements of the analytical model are:
    - dish, optionally blocked
    - Gaussian taper, default is -12dB at the edge
    - Offset to pointing centre (optional)
    - zernikes in a list of dictionaries. Each list element is of the form {"coeff":0.1, "noll":5}. See aotools for
    more details
    - Output image can be in RA, DEC coordinates or AZELGEO coordinates (the default). use_local=True means to use
    AZELGEO coordinates centered on 0deg 0deg.

    The dish is zero padded according to padding and FFT'ed to get the voltage pattern.

    :param model:
    :param pointingcentre: SkyCoord of desired pointing centre
    :param diameter: Diameter of dish in metres
    :param blockage: Blockage of dish in metres
    :param taper: "Gaussian" or None
    :param edge: Value of taper at the end of the dish (default corresponds to -12dB)
    :param zernikes: Zernikes to be applied as phase across the dish (see above)
    :param padding: Pad the image by this amount
    :param use_local: Use local frame (AZELGEO)?
    :return:
    """
    beam = create_empty_image_like(model)
    nchan, npol, ny, nx = beam["pixels"].data.shape
    padded_shape = [nchan, npol, padding * ny, padding * nx]
    padded_beam = pad_image(beam, padded_shape)
    padded_beam["pixels"].data = numpy.zeros(
        padded_beam["pixels"].data.shape, dtype="complex"
    )
    _, _, pny, pnx = padded_beam["pixels"].data.shape

    xfr = fft_image_to_griddata(padded_beam)
    cx, cy = (
        xfr.griddata_acc.griddata_wcs.sub(2).wcs.crpix[0] - 1,
        xfr.griddata_acc.griddata_wcs.sub(2).wcs.crpix[1] - 1,
    )

    for chan in range(nchan):

        # The frequency axis is the second to last in the beam
        frequency = xfr.griddata_acc.griddata_wcs.sub(["spectral"]).wcs_pix2world(
            [chan], 0
        )[0]
        wavelength = phyconst.c_m_s / frequency

        scalex = xfr.griddata_acc.griddata_wcs.sub(2).wcs.cdelt[0] * wavelength
        scaley = xfr.griddata_acc.griddata_wcs.sub(2).wcs.cdelt[1] * wavelength
        # xx, yy in metres
        xx, yy = numpy.meshgrid(
            scalex * (numpy.arange(pnx) - cx), scaley * (numpy.arange(pny) - cy)
        )

        # rr in metres
        rr = numpy.sqrt(xx ** 2 + yy ** 2)
        if beam.image_acc.polarisation_frame == PolarisationFrame("linear"):
            pols = [0, 3]
        elif beam.image_acc.polarisation_frame == PolarisationFrame("circular"):
            pols = [0, 3]
        else:
            pols = range(npol)

        combined = tapered_disk(
            rr, diameter / 2.0, blockage=blockage / 2.0, edge=edge, taper=taper
        )

        for pol in pols:
            xfr["pixels"].data[chan, pol, ...] = combined

        if pointingcentre is not None:
            # Correct for pointing centre
            pcx, pcy = pointingcentre.to_pixel(padded_beam.image_acc.wcs, origin=0)
            pxx, pyy = numpy.meshgrid(
                (numpy.arange(pnx) - cx), (numpy.arange(pny) - cy)
            )
            phase = (
                2
                * numpy.pi
                * ((pcx - cx) * pxx / float(pnx) + (pcy - cy) * pyy / float(pny))
            )
            for pol in range(npol):
                xfr["pixels"].data[chan, pol, ...] *= numpy.exp(1j * phase)

        if isinstance(zernikes, collections.abc.Iterable):
            ndisk = numpy.ceil(numpy.abs(diameter / scalex)).astype("int")[0]
            ndisk = 2 * ((ndisk + 1) // 2)
            phase = numpy.zeros([ndisk, ndisk])
            for zernike in zernikes:
                phase = zernike["coeff"] * zernike_noll(zernike["noll"], ndisk)

            # import matplotlib.pyplot as plt
            # plt.clf()
            # plt.imshow(phase)
            # plt.colorbar()
            # plt.show()
            #
            blc = pnx // 2 - ndisk // 2
            trc = pnx // 2 + ndisk // 2
            for pol in range(npol):
                xfr["pixels"].data[chan, pol, blc:trc, blc:trc] = xfr["pixels"].data[
                    chan, pol, blc:trc, blc:trc
                ] * numpy.exp(1j * phase)

    padded_beam = ifft_griddata_to_image(xfr, padded_beam)

    # Undo padding
    beam_data = padded_beam["pixels"].data[
        ...,
        (pny // 2 - ny // 2) : (pny // 2 + ny // 2),
        (pnx // 2 - nx // 2) : (pnx // 2 + nx // 2),
    ]
    for chan in range(nchan):
        beam_data[chan, ...] /= numpy.max(numpy.abs(beam_data[chan, ...]))

    beam = create_image_from_array(
        beam_data,
        wcs=beam.image_acc.wcs,
        polarisation_frame=beam.image_acc.polarisation_frame,
    )

    beam = set_pb_header(beam, use_local=use_local)

    return beam


def create_low_test_beam(model: Image, use_local=True, azel=None) -> Image:
    """Create a test power beam for LOW

    This uses an approximation that ignores the antennas

    :param model: Template image
    :param use_local: Use az el coordinates instead of ra dec
    :param azel: Tuple (Azimuth, Elevation) radians
    """
    beam = create_low_test_vp(model, use_local=use_local, azel=azel)
    beam["pixels"].data = numpy.real(
        beam["pixels"].data * numpy.conjugate(beam["pixels"].data)
    )

    set_pb_header(beam, use_local=use_local)
    return beam


def create_mid_allsky(frequency=numpy.array([1.0e9]), npixel=512, cellsize=None):
    """Approximate all sky MID beam

    Unlocked 15m dish with no taper. Actual sidelobes are likely to be lower than this model implies.

    :param frequency: Frequencies to use array(float) (Hz) default is [1e9]
    :param npixel: Number of pixels per axis (int) Default is 512
    :param cellsize: Cellsize in radians. Default is pi/npixel
    :return: Image
    """

    if not (
        isinstance(frequency, numpy.ndarray) or isinstance(frequency, xarray.DataArray)
    ):
        raise ValueError("frequency must be an array")

    if cellsize is None:
        cellsize = numpy.pi / npixel

    if len(frequency) > 1:
        channel_bandwidth = numpy.array(len(frequency) * [frequency[1] - frequency[0]])
    else:
        channel_bandwidth = numpy.array([1e6])

    # Phase centre will be overridden
    phasecentre = SkyCoord(
        ra=+0.0 * u.deg, dec=-90.0 * u.deg, frame="icrs", equinox="J2000"
    )

    vp = create_image(
        npixel=npixel,
        cellsize=cellsize,
        frequency=frequency,
        channel_bandwidth=channel_bandwidth,
        phasecentre=phasecentre,
        polarisation_frame=PolarisationFrame("linear"),
    )
    return create_vp(vp, "MID", use_local=True)


def create_low_test_vp(model: Image, use_local=True, azel=None) -> Image:
    """Create a test voltage beam for LOW

    This uses an approximation that ignores the antennas

    :param model: Template image
    :param use_local: Use az el coordinates instead of ra dec
    :param azel: Tuple (Azimuth, Elevation) radians
    :return: Image
    """
    vp_zenith = create_vp_generic(
        model, diameter=38.0, blockage=0.0, use_local=use_local
    )
    vp_zenith = set_pb_header(vp_zenith, use_local=use_local)
    if azel is None:
        return vp_zenith
    else:
        return scale_and_rotate_image(
            vp_zenith, scale=[numpy.sin(azel[1]), 1.0], angle=azel[0]
        )


def convert_azelvp_to_radec(vp, im, pa):
    """Convert AZELGEO image to image coords at specific parallactic angle

    :param pb: Primary beam or voltage pattern
    :param im: Template image
    :param pa: Parallactic angle (radians)
    :return:
    """
    vp = scale_and_rotate_image(vp, angle=pa)
    assert numpy.max(
        numpy.abs(vp["pixels"])
    ), "Scale and rotate failed: empty image {}".format(vp)

    vp_wcs = vp.image_acc.wcs
    vp_wcs.wcs.crval[0] = im.image_acc.wcs.wcs.crval[0]
    vp_wcs.wcs.crval[1] = im.image_acc.wcs.wcs.crval[1]
    vp_wcs.wcs.ctype[0] = im.image_acc.wcs.wcs.ctype[0]
    vp_wcs.wcs.ctype[1] = im.image_acc.wcs.wcs.ctype[1]

    vp = create_image_from_array(
        vp["pixels"].data, vp_wcs, vp.image_acc.polarisation_frame
    )

    rvp, footprint = reproject_image(
        vp, im.image_acc.wcs, shape=im["pixels"].data.shape
    )
    rvp["pixels"].data[footprint["pixels"].data < 1e-6] = 0.0
    assert numpy.max(
        numpy.abs(rvp["pixels"])
    ), "Reprojection failed: empty image {}".format(rvp)

    return rvp


def normalise_vp(vp):
    """Normalise the vp in place so that the peak gain on axis for parallel pols is equal

    :param vp:
    :return:
    """
    g = numpy.zeros([4])
    g[0] = numpy.max(numpy.abs(vp["pixels"].data[:, 0, ...]))
    g[3] = numpy.max(numpy.abs(vp["pixels"].data[:, 3, ...]))
    g[1] = g[2] = numpy.sqrt(g[0] * g[3])
    for chan in range(4):
        if g[chan] > 0.0:
            vp["pixels"].data[:, chan, ...] /= g[chan]
    return vp
