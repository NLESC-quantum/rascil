# This script makes a fake data set and then deconvolves it.


import logging
import sys

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models import PolarisationFrame, rascil_path
from rascil.processing_components import (
    create_blockvisibility,
    export_image_to_fits,
    deconvolve_cube,
    restore_cube,
    create_named_configuration,
    create_test_image,
    create_image_from_visibility,
    advise_wide_field,
)
from rascil.processing_components.imaging.imaging import (
    predict_blockvisibility,
    invert_blockvisibility,
)

log = logging.getLogger("rascil-logger")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))

results_dir = "./"

# Construct LOW core configuration
lowr3 = create_named_configuration("LOWBD2", rmax=750.0)

# We create the visibility. This just makes the uvw, time, antenna1, antenna2,
# weight columns in a table. We subsequently fill the visibility value in by
# a predict step.

times = numpy.zeros([1])
frequency = numpy.array([1e8])
channel_bandwidth = numpy.array([1e6])
phasecentre = SkyCoord(
    ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000"
)
vt = create_blockvisibility(
    lowr3,
    times,
    frequency,
    channel_bandwidth=channel_bandwidth,
    weight=1.0,
    phasecentre=phasecentre,
    polarisation_frame=PolarisationFrame("stokesI"),
)

# Find the recommended imaging parameters
advice = advise_wide_field(
    vt, guard_band_image=3.0, delA=0.1, oversampling_synthesised_beam=4.0
)
cellsize = advice["cellsize"]

# Read the venerable test image, constructing a RASCIL Image
m31image = create_test_image(
    cellsize=cellsize, frequency=frequency, phasecentre=vt.phasecentre
)

# Predict the visibility for the Image
vt = predict_blockvisibility(vt, m31image, context="2d")

# Make the dirty image and point spread function
model = create_image_from_visibility(vt, cellsize=cellsize, npixel=512)
dirty, sumwt = invert_blockvisibility(vt, model, context="2d")
psf, sumwt = invert_blockvisibility(vt, model, context="2d", dopsf=True)

print(
    "Max, min in dirty image = %.6f, %.6f, sumwt = %f"
    % (dirty["pixels"].data.max(), dirty["pixels"].data.min(), sumwt)
)
print(
    "Max, min in PSF         = %.6f, %.6f, sumwt = %f"
    % (psf["pixels"].data.max(), psf["pixels"].data.min(), sumwt)
)

export_image_to_fits(dirty, "%s/imaging_dirty.fits" % (results_dir))
export_image_to_fits(psf, "%s/imaging_psf.fits" % (results_dir))

# Deconvolve using clean
comp, residual = deconvolve_cube(
    dirty,
    psf,
    niter=10000,
    threshold=0.001,
    fractional_threshold=0.001,
    window_shape="quarter",
    gain=0.7,
    scales=[0, 3, 10, 30],
)

restored = restore_cube(comp, psf, residual)
print(
    "Max, min in restored image = %.6f, %.6f, sumwt = %f"
    % (restored["pixels"].data.max(), restored["pixels"].data.min(), sumwt)
)
export_image_to_fits(restored, "%s/imaging_restored.fits" % (results_dir))
