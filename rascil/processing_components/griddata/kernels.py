"""
Functions that define and manipulate kernels

"""

__all__ = [
    "create_pswf_convolutionfunction",
    "create_box_convolutionfunction",
    "create_awterm_convolutionfunction",
    "create_vpterm_convolutionfunction",
]

import logging

import numpy
from astropy.wcs import WCS

from rascil.data_models import Image
from rascil.processing_components.fourier_transforms.fft_coordinates import (
    coordinates,
    grdsf,
)
from rascil.processing_components.griddata.convolution_functions import (
    create_convolutionfunction_from_image,
)
from rascil.processing_components.image.operations import (
    create_image_from_array,
    create_empty_image_like,
    fft_image_to_griddata,
    pad_image,
    create_w_term_like,
)
from rascil.processing_components.imaging.primary_beams import convert_azelvp_to_radec

log = logging.getLogger("rascil-logger")


def create_box_convolutionfunction(
    im, oversampling=1, support=1, polarisation_frame=None
):
    """Fill a box car function into a ConvolutionFunction

    Also returns the griddata correction function as an image

    :param im: Image template
    :param oversampling: Oversampling of the convolution function in uv space
    :return: griddata correction Image, griddata kernel as ConvolutionFunction
    """
    ##assert isinstance(im, Image)
    cf = create_convolutionfunction_from_image(
        im, oversampling=1, support=4, polarisation_frame=polarisation_frame
    )

    nchan, npol, _, _ = im["pixels"].data.shape

    cf["pixels"].data[...] = 0.0 + 0.0j
    cf["pixels"].data[..., 2, 2] = 1.0 + 0.0j

    # Now calculate the griddata correction function as an image with the same coordinates as the image
    # which is necessary so that the correction function can be applied directly to the image
    nchan, npol, ny, nx = im["pixels"].data.shape
    nu = numpy.abs(coordinates(nx))

    gcf1d = numpy.sinc(nu)
    gcf = numpy.outer(gcf1d, gcf1d)
    gcf = 1.0 / gcf

    gcf_data = numpy.zeros_like(im["pixels"].data)
    gcf_data[...] = gcf[numpy.newaxis, numpy.newaxis, ...]
    gcf_image = create_image_from_array(
        gcf_data, im.image_acc.wcs, im.image_acc.polarisation_frame
    )

    assert cf["pixels"].data.dtype == "complex", cf["pixels"].data.dtype
    assert gcf_image["pixels"].data.dtype == "float", gcf_image["pixels"].data.dtype
    return gcf_image, cf


def create_pswf_convolutionfunction(
    im, oversampling=127, support=8, polarisation_frame=None
):
    """Fill an Anti-Aliasing filter into a ConvolutionFunction

    Fill the Prolate Spheroidal Wave Function into a GriData with the specified oversampling. Only the inner
    non-zero part is retained

    Also returns the griddata correction function as an image

    :param im: Image template
    :param oversampling: Oversampling of the convolution function in uv space
    :return: griddata correction Image, griddata kernel as ConvolutionFunction
    """
    ##assert isinstance(im, Image), im
    if oversampling % 2 == 0:
        oversampling += 1
        log.info(
            "Setting oversampling to next greatest odd number {}".format(oversampling)
        )

    width = support - 2
    # Calculate the convolution kernel. We oversample in u,v space by the factor oversampling
    cf = create_convolutionfunction_from_image(
        im,
        oversampling=oversampling,
        support=support,
        polarisation_frame=polarisation_frame,
    )

    kernel = numpy.zeros([oversampling, support])
    for grid in range(1, support - 1):
        for subsample in range(oversampling):
            nu = (grid - support // 2) - (subsample - oversampling // 2) / oversampling
            kernel[subsample, grid] = grdsf([nu / (width // 2)])[1]

    kernel /= numpy.sum(numpy.real(kernel[oversampling // 2, :]))

    nchan, npol, _, _ = im["pixels"].data.shape

    cf["pixels"].data = numpy.zeros(
        [nchan, npol, 1, oversampling, oversampling, support, support]
    ).astype("complex")
    for y in range(oversampling):
        for x in range(oversampling):
            cf["pixels"].data[:, :, 0, y, x, :, :] = numpy.outer(
                kernel[y, :], kernel[x, :]
            )[numpy.newaxis, numpy.newaxis, ...]

    for y in range(oversampling):
        for x in range(oversampling):
            norm = numpy.sum(
                numpy.real(cf["pixels"].data[:, :, 0, y, x, :, :]), axis=(-2, -1)
            )[..., numpy.newaxis, numpy.newaxis]
            cf["pixels"].data[:, :, 0, y, x, :, :] /= norm

    # Now calculate the griddata correction function as an image with the same coordinates as the image
    # which is necessary so that the correction function can be applied directly to the image
    nchan, npol, ny, nx = im["pixels"].data.shape
    nu = numpy.abs(2.0 * coordinates(nx))
    gcf1d = grdsf(nu)[0]
    gcf = numpy.outer(gcf1d, gcf1d)
    gcf[gcf > 0.0] = gcf.max() / gcf[gcf > 0.0]

    gcf_data = numpy.zeros_like(im["pixels"].data)
    gcf_data[...] = gcf[numpy.newaxis, numpy.newaxis, ...]
    gcf_image = create_image_from_array(
        gcf_data, im.image_acc.wcs, im.image_acc.polarisation_frame
    )

    assert cf["pixels"].data.dtype == "complex", cf["pixels"].data.dtype
    assert gcf_image["pixels"].data.dtype == im["pixels"].data.dtype, gcf[
        "pixels"
    ].data.dtype
    return gcf_image, cf


def create_awterm_convolutionfunction(
    im,
    make_pb=None,
    nw=1,
    wstep=1e15,
    oversampling=9,
    support=8,
    use_aaf=True,
    maxsupport=512,
    pa=None,
    normalise=True,
    polarisation_frame=None,
):
    """Fill AW projection kernel into a GridData.

    :param im: Image template
    :param make_pb: Function to make the primary beam model image (hint: use a partial)
    :param nw: Number of w planes
    :param wstep: Step in w (wavelengths)
    :param oversampling: Oversampling of the convolution function in uv space
    :return: griddata correction Image, griddata kernel as GridData
    """
    if oversampling % 2 == 0:
        oversampling += 1
        log.info(
            "Setting oversampling to next greatest odd number {}".format(oversampling)
        )

    d2r = numpy.pi / 180.0

    # We only need the griddata correction function for the PSWF so we make
    # it for the shape of the image
    nchan, npol, ony, onx = im["pixels"].data.shape

    # assert isinstance(im, Image)
    # Calculate the template convolution kernel.
    cf = create_convolutionfunction_from_image(
        im,
        nw=nw,
        wstep=wstep,
        oversampling=oversampling,
        support=support,
        polarisation_frame=polarisation_frame,
    )

    if nw > 1 and numpy.abs(wstep) > 0.0:
        w_list = cf.convolutionfunction_acc.cf_wcs.sub([5]).wcs_pix2world(range(nw), 0)[
            0
        ]
    else:
        w_list = [0.0]

    # assert isinstance(oversampling, int)
    assert oversampling > 0

    nx = max(maxsupport, 2 * oversampling * support)
    ny = max(maxsupport, 2 * oversampling * support)

    qnx = nx // oversampling
    qny = ny // oversampling

    cf["pixels"].data[...] = 0.0

    ccell = onx * numpy.abs(d2r * im.image_acc.wcs.wcs.cdelt[0]) / qnx

    subim_wcs = im.image_acc.wcs.deepcopy()
    subim_wcs.wcs.cdelt[0] = -ccell / d2r
    subim_wcs.wcs.cdelt[1] = +ccell / d2r
    subim_wcs.wcs.crpix[0] = qnx // 2 + 1.0
    subim_wcs.wcs.crpix[1] = qny // 2 + 1.0
    subim = create_image_from_array(
        numpy.zeros([nchan, npol, qny, qnx]),
        subim_wcs,
        polarisation_frame=polarisation_frame,
    )

    if use_aaf:
        this_pswf_gcf, _ = create_pswf_convolutionfunction(
            subim, oversampling=1, support=6, polarisation_frame=polarisation_frame
        )
        norm = 1.0 / this_pswf_gcf["pixels"].data
    else:
        norm = 1.0

    if make_pb is not None:
        pb = make_pb(subim)

        if pa is not None:
            rpb = convert_azelvp_to_radec(pb, subim, pa)
        else:
            rpb = convert_azelvp_to_radec(pb, subim, 0.0)

        norm *= rpb["pixels"].data

    # We might need to work with a larger image
    padded_shape = [nchan, npol, ny, nx]
    thisplane = subim.copy(deep=True)
    thisplane["pixels"].data = numpy.zeros(
        thisplane["pixels"].data.shape, dtype="complex"
    )
    scf = cf["pixels"].data[...]
    for z, w in enumerate(w_list):
        thisplane["pixels"].data[...] = 0.0 + 0.0j
        thisplane = create_w_term_like(thisplane, w, dopol=True)
        thisplane["pixels"].data *= norm
        paddedplane = pad_image(thisplane, padded_shape)
        grid = fft_image_to_griddata(paddedplane)
        gd = grid["pixels"].data

        ycen, xcen = ny // 2, nx // 2
        for y in range(oversampling):
            ybeg = y + ycen + (support * oversampling) // 2 - oversampling // 2
            yend = y + ycen - (support * oversampling) // 2 - oversampling // 2
            assert ybeg >= 0, "Convolutionfunction does not fit into image"
            assert yend < ny, "Convolutionfunction does not fit into image"
            for x in range(oversampling):
                xbeg = x + xcen + (support * oversampling) // 2 - oversampling // 2
                xend = x + xcen - (support * oversampling) // 2 - oversampling // 2
                assert xbeg >= 0, "Convolutionfunction does not fit into image"
                assert xend < nx, "Convolutionfunction does not fit into image"
                scf[..., z, y, x, :, :] = gd[
                    ..., ybeg:yend:-oversampling, xbeg:xend:-oversampling
                ]

    cf["pixels"].data[...] = scf

    for ipol in range(1, npol):
        cf["pixels"].data[:, ipol, ...] = cf["pixels"].data[:, 0, ...]

    if normalise:
        norm = numpy.zeros([nchan, npol, oversampling, oversampling])
        for y in range(oversampling):
            for x in range(oversampling):
                norm[..., y, x] = numpy.sum(
                    numpy.real(cf["pixels"].data[:, :, 0, y, x, :, :]), axis=(-2, -1)
                )
        for z, _ in enumerate(w_list):
            for y in range(oversampling):
                for x in range(oversampling):
                    cf["pixels"].data[:, :, z, y, x] /= norm[..., y, x][
                        ..., numpy.newaxis, numpy.newaxis
                    ]
    cf["pixels"].data = numpy.conjugate(cf["pixels"].data)

    if use_aaf:
        pswf_gcf, _ = create_pswf_convolutionfunction(
            im, oversampling=1, support=6, polarisation_frame=polarisation_frame
        )
    else:
        pswf_gcf = create_empty_image_like(im)
        pswf_gcf["pixels"].data[...] = 1.0

    assert cf["pixels"].data.dtype == "complex", cf["pixels"].data.dtype
    assert pswf_gcf["pixels"].data.dtype == im["pixels"].data.dtype, pswf_gcf[
        "pixels"
    ].data.dtype
    return pswf_gcf, cf


def create_vpterm_convolutionfunction(
    im,
    make_vp=None,
    oversampling=8,
    support=6,
    use_aaf=False,
    maxsupport=512,
    pa=None,
    normalise=True,
    polarisation_frame=None,
):
    """Fill voltage pattern kernel projection kernel into a GridData.

    The makes the convolution function for gridding polarised data with a voltage
    pattern.

    :param im: Image template
    :param make_vp: Function to make the voltage pattern model image (hint: use a partial)
    :param oversampling: Oversampling of the convolution function in uv space
    :return: griddata correction Image, griddata kernel as GridData
    """
    if oversampling % 2 == 0:
        log.info(
            "Setting oversampling to next greatest odd number {}".format(oversampling)
        )
        oversampling += 1

    d2r = numpy.pi / 180.0

    # We only need the griddata correction function for the PSWF so we make
    # it for the shape of the image
    nchan, npol, ony, onx = im["pixels"].data.shape

    # assert isinstance(im, Image)
    # Calculate the template convolution kernel.
    cf = create_convolutionfunction_from_image(
        im,
        oversampling=oversampling,
        support=support,
        polarisation_frame=polarisation_frame,
    )

    cf_shape = list(cf["pixels"].data.shape)
    cf["pixels"].data = numpy.zeros(cf_shape).astype("complex")

    # assert isinstance(oversampling, int)
    assert oversampling > 0

    nx = max(maxsupport, 2 * oversampling * support)
    ny = max(maxsupport, 2 * oversampling * support)

    qnx = nx // oversampling
    qny = ny // oversampling

    cf["pixels"].data[...] = 0.0

    ccell = onx * numpy.abs(d2r * im.image_acc.wcs.wcs.cdelt[0]) / qnx

    wcs = im.image_acc.wcs
    wcs.wcs.cdelt[0] = -ccell / d2r
    wcs.wcs.cdelt[1] = +ccell / d2r
    wcs.wcs.crpix[0] = qnx // 2 + 1.0
    wcs.wcs.crpix[1] = qny // 2 + 1.0

    subim = create_image_from_array(
        numpy.zeros([nchan, npol, qny, qnx]),
        wcs=wcs,
        polarisation_frame=im.image_acc.polarisation_frame,
    )

    vp = make_vp(subim)
    assert (
        numpy.max(numpy.abs(vp["pixels"])) > 0.0
    ), "Construction of voltage pattern failed: empty image {}".format(vp)

    if pa is not None:
        rvp = convert_azelvp_to_radec(vp, subim, pa)
    else:
        rvp = convert_azelvp_to_radec(vp, subim, 0.0)

    assert (
        numpy.max(numpy.abs(rvp["pixels"])) > 0.0
    ), "Projection from azel to radec failed: empty image {}".format(rvp)

    if use_aaf:
        this_pswf_gcf, _ = create_pswf_convolutionfunction(
            subim, oversampling=1, support=6, polarisation_frame=polarisation_frame
        )
        rvp["pixels"].data /= this_pswf_gcf["pixels"].data

    # We might need to work with a larger image
    padded_shape = [nchan, npol, ny, nx]
    paddedplane = pad_image(rvp, padded_shape)
    paddedplane = fft_image_to_griddata(paddedplane)

    ycen, xcen = ny // 2, nx // 2
    for y in range(oversampling):
        ybeg = y + ycen + (support * oversampling) // 2 - oversampling // 2
        yend = y + ycen - (support * oversampling) // 2 - oversampling // 2
        # vv = range(ybeg, yend, -oversampling)
        for x in range(oversampling):
            xbeg = x + xcen + (support * oversampling) // 2 - oversampling // 2
            xend = x + xcen - (support * oversampling) // 2 - oversampling // 2

            # uu = range(xbeg, xend, -oversampling)
            cf["pixels"].data[..., 0, y, x, :, :] = paddedplane["pixels"].data[
                ..., ybeg:yend:-oversampling, xbeg:xend:-oversampling
            ]

    if normalise:
        cf["pixels"].data /= numpy.sum(
            numpy.real(
                cf["pixels"].data[0, 0, 0, oversampling // 2, oversampling // 2, :, :]
            )
        )
    cf["pixels"].data = numpy.conjugate(cf["pixels"].data)

    if use_aaf:
        pswf_gcf, _ = create_pswf_convolutionfunction(
            im, oversampling=1, support=6, polarisation_frame=polarisation_frame
        )
    else:
        pswf_gcf = create_empty_image_like(im)
        pswf_gcf["pixels"].data[...] = 1.0

    assert cf["pixels"].data.dtype == "complex", cf["pixels"].data.dtype
    assert pswf_gcf["pixels"].data.dtype == "float", pswf_gcf["pixels"].data.dtype
    return pswf_gcf, cf
