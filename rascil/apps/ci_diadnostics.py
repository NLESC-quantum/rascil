# coding: utf-8
""" Power spectrum for an image, adapted from Fred Dulwich code
"""
import os
import logging
import sys

from scipy import optimize
import numpy as np
import astropy.constants as consts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy

from rascil.processing_components import fft_image_to_griddata, import_image_from_fits, show_image
from rascil.data_models.xarray_coordinate_support import griddata_wcs

log = logging.getLogger("rascil-logger")


def bdsf_qa_image(im_data):
    """Assess the quality of an image

    Set of statistics of an image: max, min, maxabs, rms, sum, medianabs,
    medianabsdevmedian, median and mean.

    :param im_data: image data
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

    log.info("QA of pybdsf image:")
    for item in image_stats.items():
        log.info("    {}".format(item))

    return image_stats


def gaussian(x, amplitude, mean, stddev):
    """
    Gaussian fit for histogram.

    :param x: x-axis points
    :param amplitude: gaussian applitude
    :param mean: mean on the distribution
    :param stddev: standard deviation of the distribution

    :return Gaussian finction
    """
    return amplitude * np.exp(-(((x - mean) / 4.0 / stddev) ** 2))


def histogram(bdsf_image, input_image, image_type):
    """
    Plot a histogram of the pixel counts produced with
    mean, RMS and Gaussian fit.

    :param bdsf_image: pybdsf image object
    :param input_image_residual: file name of input image
    :param image_type: type of imput image; either restored or residual

    :return None
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

    # Add shaded region of withth 2*RMS centered on the mean.
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
    ax.set_xlabel("Value")
    ax.legend()

    # Create histogram file name from input image name removeing file extension.
    save_hist = (
        input_image.replace(".fits" if ".fits" in input_image else ".h5", "")
        + "_"
        + image_type
        + "_gaus_hist"
    )

    ax.set_title(image_type)

    plt.tight_layout()

    log.info('Saving histogram to "{}.png"'.format(save_hist))
    plt.savefig(save_hist + ".png")
    plt.close()

    return


def running_mean(bdsf_image):
    """
    Image plot and running mean.
    """

    log.info("Plotting sky image with running mean.")

    return


def source_region_mask(img):
    """
    Mask pixels from an image which are within 5*beam_width of sources in the
    source catalogue.
    """
    # Here the major axis of the beam is used as the beam width and the. pybdsf
    # gives the beam "IN SIGMA UNITS in pixels" so we need to convert to
    # straight pixels by multiplying by the FWHM. See init_beam() function in
    # pybdsf source code.
    beam_width = img.pixel_beam[0]*2.35482
    beam_radius = beam_width/2.0

    # x_extent = img.

    # Xposn: the x image coordinate of the source, in pixels
    # Yposn: the y image coordinate of the source, in pixels

    for source in source_list:
        x_source =  Yposn
        source_radius = np.sqrt(x_pix*x_pix + y_pix*y_pix)

    histogram(im_data, input_image, image_type)

    return


def power_spectrum(image, signal_channel, noise_channel, resolution):
    """
    Plot power spectrum for an image.

    :param image: image object
    :param signal_channel: channel containing both signal and noise
    :param noise_channel: containing noise only
    :param resolution: Resolution in radians needed for conversion to K

    :return None
    """

    basename = os.path.basename(os.getcwd())

    print("Display power spectrum of an image")

    im = import_image_from_fits(image)

    nchan, npol, ny, nx = im["pixels"].shape

    if signal_channel is None:
        signal_channel = nchan // 2

    plt.clf()
    show_image(im, chan=signal_channel)
    plt.title('Signal image %s' % (basename))
    plt.savefig('simulation_image_channel_%d.png' % signal_channel)
    plt.show()
    plt.clf()
    show_image(im, chan=noise_channel)
    plt.title('Noise image %s' % (basename))
    plt.savefig('simulation_noise_channel_%d.png' % signal_channel)
    plt.show()

    print(im)
    imfft = fft_image_to_griddata(im)
    print(imfft)

    omega = numpy.pi * resolution ** 2 / (4 * numpy.log(2.0))
    wavelength = consts.c / numpy.average(im.frequency)
    kperjy = 1e-26 * wavelength ** 2 / (2 * consts.k_B * omega)

    im_spectrum = imfft.copy()
    im_spectrum["pixels"].data = kperjy.value * numpy.abs(imfft["pixels"].data)
    noisy = numpy.max(im_spectrum["pixels"].data[noise_channel, 0]) > 0.0

    profile = radial_profile(im_spectrum["pixels"].data[signal_channel, 0])
    noise_profile = radial_profile(im_spectrum["pixels"].data[noise_channel, 0])

    plt.clf()
    cellsize_uv = numpy.abs(griddata_wcs(imfft).wcs.cdelt[0])
    lambda_max = cellsize_uv * len(profile)
    lambda_axis = numpy.linspace(cellsize_uv, lambda_max, len(profile))
    theta_axis = 180.0 / (numpy.pi * lambda_axis)
    plt.plot(theta_axis, profile, color='blue', label='signal')
    if noisy:
        plt.plot(theta_axis, noise_profile, color='red', label='noise')
    plt.gca().set_title("Power spectrum of image %s" % (basename))
    plt.gca().legend()
    plt.gca().set_xlabel(r"$\theta$")
    plt.gca().set_ylabel(r"$K^2$")
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.gca().set_ylim(1e-6 * numpy.max(profile), 2.0 * numpy.max(profile))
    plt.tight_layout()
    plt.savefig('power_spectrum_profile_channel_%d.png' % signal_channel)
    plt.show()

    filename = 'power_spectrum_channel.csv'
    results = list()
    for row in range(len(theta_axis)):
        result = dict()
        result['inverse_theta'] = theta_axis[row]
        result['profile'] = profile[row]
        result['noise_profile'] = noise_profile[row]
        results.append(result)

    import csv
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results[0].keys(), delimiter=',', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
        csvfile.close()

    return


def main():
    qa_image(im_data)
    histogram(bdsf_image, input_image, image_type)
    power_spectrum(image, signal_channel, noise_channel, resolution)


if __name__ == "__main__":
    main()
