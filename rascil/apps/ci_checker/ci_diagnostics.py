"""
Collection of diagnostic function for use with the RASCIL continuum imaging
checker.
"""

import logging
import csv

from astropy.wcs.wcsapi import SlicedLowLevelWCS
from scipy import optimize
import numpy as np
import astropy.constants as consts

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy

from rascil.processing_components import fft_image_to_griddata, import_image_from_fits
from rascil.data_models.xarray_coordinate_support import griddata_wcs

log = logging.getLogger("rascil-logger")


def qa_image_bdsf(im_data, description="image"):
    """Assess the quality of an image

    Set of statistics of an image: max, min, maxabs, rms, sum, medianabs,
    medianabsdevmedian, median and mean.

    :param im_data: image data, either numpy Array or numpy MaskedArray
    :param description: string describing array
    :return image_stats: statistics of the image
    """

    image_stats = {
        "shape": str(im_data.data.shape),
        "max": np.max(im_data),
        "min": np.min(im_data),
        "maxabs": np.max(np.abs(im_data)),
        "rms": np.std(im_data),
        "sum": np.sum(im_data),
        "medianabs": np.median(np.abs(im_data)),
        "medianabsdevmedian": np.median(np.abs(im_data - np.median(im_data))),
        "median": np.median(im_data),
        "mean": np.mean(im_data),
    }

    log.info(f"QA of {description}:")
    for item in image_stats.items():
        log.info("    {}".format(item))

    return image_stats


def plot_name(input_image, image_type, plot_type):
    """
    Create file name from input image name removing file extension.

    :param input_image: File name of input image
    :param image_type: Type of image e.g. restored
    :param plot_type: type of plot (e.g. hist, plot...)

    :return: file name for saved plot
    """
    return (
        input_image.replace(".fits" if ".fits" in input_image else ".h5", "")
        + "_"
        + image_type
        + "_"
        + plot_type
    )


def gaussian(x, amplitude, mean, stddev):
    """
    Gaussian fit for histogram.

    :param x: x-axis points
    :param amplitude: gaussian amplitude
    :param mean: mean on the distribution
    :param stddev: standard deviation of the distribution

    :return Gaussian function
    """

    gauss = amplitude * np.exp(-1.0 / 2.0 * ((x - mean) / stddev) ** 2)
    return gauss


def histogram(bdsf_image, input_image, description="image"):
    """
    Plot a histogram of the pixel counts produced with
    mean, RMS and Gaussian fit.

    :param bdsf_image: pybdsf image object
    :param input_image_residual: file name of input image
    :param description: type of input image
    """

    im_data = bdsf_image.resid_gaus_arr

    fig, ax = plt.subplots()

    counts, bins, _ = ax.hist(
        im_data.ravel(), bins=1000, density=False, zorder=5, histtype="step"
    )

    # "bins" are the bin edge points, so need the mid points.
    mid_points = bins[:-1] + 0.5 * abs(bins[1:] - bins[:-1])

    p0 = [counts.max(), bdsf_image.raw_mean, bdsf_image.raw_rms]

    popt, pcov = optimize.curve_fit(gaussian, mid_points, counts, p0=p0)
    mean = popt[1]
    stddev = abs(popt[2])

    ax.plot(mid_points, gaussian(mid_points, *popt), label="fit", zorder=10)
    ax.axvline(
        popt[1], color="C2", linestyle="--", label=f"mean: {mean:.3e}", zorder=15
    )

    # Add shaded region of width 2*RMS centered on the mean.
    rms_region = [mean - stddev, mean + stddev]
    ax.axvspan(
        rms_region[0],
        rms_region[1],
        facecolor="C2",
        alpha=0.3,
        zorder=10,
        label=f"stddev: {stddev:.3e}",
    )

    ax.set_yscale("symlog")
    ax.set_ylabel("Counts")
    ax.set_xlabel(r"Flux $\left( \rm{Jy/\rm{beam}} \right) $")
    ax.legend()

    save_plot = plot_name(input_image, description, "hist")

    ax.set_title(description)

    plt.tight_layout()

    log.info('Saving histogram to "{}.png"'.format(save_plot))
    plt.savefig(save_plot + ".png")
    plt.close()


def plot_with_running_mean(img, input_image, stats, projection, description="image"):
    """
    Image plot and running mean.

    :param img: pybdsf image object
    :param input_image_residual: file name of input image
    :param stats: statistics of image
    :param projection: projection from World Coordinate System (WCS) object
    :param description: string to put in png file name and plot title
    """

    log.info("Plotting sky image with running mean.")

    try:
        image = img.image_arr[0, 0, :, :]
    except AttributeError:
        image = img

    x_index = np.arange(0, image.shape[-2])
    y_index = np.arange(0, image.shape[-1])

    fig = plt.figure(figsize=(9, 8), constrained_layout=False)
    grid = fig.add_gridspec(nrows=4, ncols=4)

    main_ax = fig.add_subplot(grid[:-1, 1:], projection=projection)

    y_plot = fig.add_subplot(grid[:-1, 0], sharey=main_ax, projection=projection)
    y_plot.set_ylabel("DEC---SIN")

    y_plot2 = y_plot.twiny()
    y_plot2.plot(np.mean(image, axis=0), y_index)
    y_plot2.set_xlabel("mean")

    x_plot = fig.add_subplot(grid[-1, 1:], sharex=main_ax, projection=projection)
    x_plot.set_xlabel("RA---SIN")

    x_plot2 = x_plot.twinx()
    x_plot2.plot(x_index, np.mean(image, axis=1))
    x_plot2.set_ylabel("mean")

    imap = main_ax.imshow(image.T, origin="lower", aspect=1)

    if description == "restored":
        for gaussian in img.gaussians:
            source = plt.Circle(
                (gaussian.centre_pix[0], gaussian.centre_pix[1]),
                color="w",
                fill=False,
            )
            main_ax.add_patch(source)
    main_ax.title.set_text("Running mean of " + description)

    main_pos = main_ax.get_position()
    dh = 0.008
    ax_cbar = fig.add_axes(
        [main_pos.x1 - 0.057, main_pos.y0 - dh, 0.015, main_pos.y1 - main_pos.y0 + dh]
    )
    plt.colorbar(imap, cax=ax_cbar, label=r"Flux $\left( \rm{Jy/\rm{beam}} \right) $")

    i = 0
    for key, val in stats.items():
        if i == 0:
            string = f"{key}: {val}"
        else:
            string = f"{key}: {val:.3e}"

        if val is not np.ma.masked:
            plt.text(
                0.02,
                0.25 - i * 0.025,
                string,
                fontsize=10,
                transform=plt.gcf().transFigure,
            )
            i += 1

    plt.subplots_adjust(wspace=0.0001, hspace=0.0001, right=0.81)

    save_plot = plot_name(input_image, description, "plot")

    log.info('Saving sky plot to "{}.png"'.format(save_plot))
    plt.savefig(save_plot + ".png", pad_inches=-1)
    plt.close()


def source_region_mask(img):
    """
    Mask pixels from an image which are within 5*beam_width of sources in the
    source catalogue.

    :param img: pybdsf image object to be masked

    :return source_mask, background_mask: copies of masked input array.
    """

    log.info("Masking source and background regions.")

    # Here the major axis of the beam is used as the beam width and the. pybdsf
    # gives the beam "IN SIGMA UNITS in pixels" so we need to convert to
    # straight pixels by multiplying by the FWHM. See init_beam() function in
    # pybdsf source code.
    beam_width = img.pixel_beam()[0] * 2.35482
    beam_radius = beam_width / 2.0

    # img.image_arr.shape --> (nstokes, nchannels, img_size_x, img_size_y)
    image_to_be_masked = img.image_arr[0, 0, :, :]

    image_shape = [image_to_be_masked.shape[-2], image_to_be_masked.shape[-1]]

    grid = np.meshgrid(
        np.arange(0, image_to_be_masked.shape[-2]),
        np.arange(0, image_to_be_masked.shape[-1]),
        sparse=True,
        indexing="ij",
    )

    source_regions = np.ones(shape=image_shape, dtype=int)
    background_regions = np.zeros(shape=image_shape, dtype=int)

    for gaussian in img.gaussians:

        source_radius = np.sqrt(
            (grid[0] - gaussian.centre_pix[0]) ** 2
            + (grid[1] - gaussian.centre_pix[1]) ** 2
        )

        source_regions[source_radius < beam_radius] = 0

    background_regions[source_regions == 0] = 1

    source_mask = np.ma.array(image_to_be_masked, mask=source_regions, copy=True)

    background_mask = np.ma.array(
        image_to_be_masked, mask=background_regions, copy=True
    )

    return source_mask, background_mask


def _radial_profile(image, centre=None):
    """
    Function for calculating the radial profile of input image.

    :param image: 2D numpy array
    :param centre: centre of the image
    """
    if centre is None:
        centre = (image.shape[0] // 2, image.shape[1] // 2)
    x, y = np.indices((image.shape[0:2]))
    r = np.sqrt((x - centre[0]) ** 2 + (y - centre[1]) ** 2)
    r = r.astype(int)
    return np.bincount(r.ravel(), image.ravel()) / np.bincount(r.ravel())


def _plot_power_spectrum(input_image, profile, theta_axis):
    """
    Plot the power spectrum and save it as PNG.

    :param input_image: name of input image (e.g. FITS file name)
    :param profile:
    :param theta_axis:
    """
    plt.clf()

    plt.plot(theta_axis, profile)
    plt.gca().set_title("Power spectrum of image residual")
    plt.gca().set_xlabel(r"$\theta$")
    plt.gca().set_ylabel(r"$K^2$")
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.gca().set_ylim(1e-6 * numpy.max(profile), 2.0 * numpy.max(profile))
    plt.tight_layout()

    power_sp_plot_name = plot_name(input_image, "residual", "power_spectrum")

    log.info('Saving power spectrum to "{}.png"'.format(power_sp_plot_name))
    plt.savefig(power_sp_plot_name + ".png")
    plt.close()

    return power_sp_plot_name


def _save_power_spectrum_to_csv(profile, theta_axis, file_name):
    """
    Save the power spectrum into CSV.

    :param profile:
    :param theta_axis:
    :param file_name: string the csv file name should contain
    """
    log.info('Saving power spectrum profile to "{}_channel.csv"'.format(file_name))
    filename = file_name + "_channel.csv"

    results = list()
    for row in range(len(theta_axis)):
        result = dict()
        result["inverse_theta"] = theta_axis[row]
        result["profile"] = profile[row]
        results.append(result)

    with open(filename, "w") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=results[0].keys(),
            delimiter=",",
            quotechar="|",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result)
        csvfile.close()


def power_spectrum(input_image, resolution, signal_channel=None):
    """
    Calculate the power spectrum of an image.

    :param input_image: FITS file to read data from
    :param resolution: Resolution in radians needed for conversion to K <-- what is K?
    :param signal_channel: channel containing both signal and noise, optional

    :return (profile, theta_axis) --> what are these?
    """

    im = import_image_from_fits(input_image)

    nchan, npol, ny, nx = im["pixels"].shape

    if signal_channel is None:
        signal_channel = nchan // 2

    imfft = fft_image_to_griddata(im)

    omega = numpy.pi * resolution ** 2 / (4 * numpy.log(2.0))
    wavelength = consts.c / numpy.average(im.frequency)
    kperjy = 1e-26 * wavelength ** 2 / (2 * consts.k_B * omega)

    im_spectrum = imfft.copy()
    im_spectrum["pixels"].data = kperjy.value * numpy.abs(imfft["pixels"].data)

    profile = _radial_profile(im_spectrum["pixels"].data[signal_channel, 0])

    cellsize_uv = numpy.abs(griddata_wcs(imfft).wcs.cdelt[0])
    lambda_max = cellsize_uv * len(profile)
    lambda_axis = numpy.linspace(cellsize_uv, lambda_max, len(profile))
    theta_axis = 180.0 / (numpy.pi * lambda_axis)

    return profile, theta_axis


def ci_checker_diagnostics(bdsf_image, input_image, image_type):
    """
    Log and plot diagnostics of an image generated by bdfs.
    A histogram of the pixel counts produced with mean, RMS and Gaussian fit.
    Along with running mean. A power spectrum of the residual is also produced.

    :param bdsf_image: pybdsf image object
    :param input_image: file name of input image
    :param image_type: type of imput image; either restored or residual
    """

    # Setting the first to slices to 0 meas we are taking the first frequency
    # and first polarisation.
    # TODO: if support for polarasation is to be added this needs to be changed.
    slices = [0, 0, slice(bdsf_image.shape[-1]), slice(bdsf_image.shape[-2])]
    subwcs = SlicedLowLevelWCS(bdsf_image.wcs_obj, slices=slices)

    log.info("Performing image diagnostics")

    if image_type == "residual":

        residual_stats = qa_image_bdsf(
            bdsf_image.image_arr[0, 0, :, :], description="residual"
        )
        plot_with_running_mean(
            bdsf_image.image_arr[0, 0, :, :],
            input_image,
            residual_stats,
            subwcs,
            description="residual",
        )
        histogram(bdsf_image, input_image, description="residual")

        # calculate, plot, and save power spectrum
        profile, theta_axis = power_spectrum(input_image, 5.0e-4)
        save_power_spectrum_plot = _plot_power_spectrum(
            input_image, profile, theta_axis
        )
        _save_power_spectrum_to_csv(profile, theta_axis, save_power_spectrum_plot)

    elif image_type == "restored":

        source_mask, background_mask = source_region_mask(bdsf_image)

        sources_stats = qa_image_bdsf(source_mask, description="sources")
        background_stats = qa_image_bdsf(background_mask, description="background")
        restored_stats = qa_image_bdsf(
            bdsf_image.image_arr[0, 0, :, :], description="restored"
        )
        plot_with_running_mean(
            source_mask, input_image, sources_stats, subwcs, description="sources"
        )
        plot_with_running_mean(
            background_mask,
            input_image,
            background_stats,
            subwcs,
            description="background",
        )
        plot_with_running_mean(
            bdsf_image, input_image, restored_stats, subwcs, description="restored"
        )
