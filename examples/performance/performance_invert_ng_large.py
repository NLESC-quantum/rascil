""" Test of performance of nifty gridder when running on a MID data set.

Typical usage:

time python examples/performance/performance_invert_ng_large.py --scale 16 --threads 64
Scale 16.0 threads 64 duration 424.721 s

real	7m13.111s
user	412m0.181s
sys	1m10.731s

"""
import logging
import sys
import argparse
import time

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models import Skycomponent
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import invert_ng
from rascil.processing_components.imaging import dft_skycomponent_visibility
from rascil.processing_components.simulation import (
    create_named_configuration,
)
from rascil.processing_components.simulation import (
    ingest_unittest_visibility,
    create_unittest_model,
)

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))


def main():
    parser = argparse.ArgumentParser(
        description="Test ng performance", fromfile_prefix_chars="@"
    )
    parser.add_argument(
        "--threads", type=int, default=16, help="Number of threads for ng"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scaling factor",
    )
    args = parser.parse_args()

    scale = int(args.scale)
    npixel = 1024 * scale
    rmax = 5000.0
    cellsize = numpy.pi / (180.0 * 3600.0)

    low = create_named_configuration("MID", rmax=rmax)
    times = numpy.arange(-4 * 3600.0, +4 * 3600, 1800.0) * numpy.pi / 43200.0

    nchan = 8
    frequency = numpy.linspace(1.3e9, 1.36e9, nchan)
    channelwidth = numpy.array(nchan * [frequency[1] - frequency[0]])

    blockvis_pol = PolarisationFrame("stokesI")
    image_pol = PolarisationFrame("stokesI")
    flux = 100.0 * numpy.ones([nchan, 1])

    phasecentre = SkyCoord(
        ra=+180.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000"
    )
    blockvis = ingest_unittest_visibility(
        low,
        frequency,
        channelwidth,
        times,
        blockvis_pol,
        phasecentre,
    )

    model = create_unittest_model(
        blockvis,
        image_pol,
        npixel=npixel,
        nchan=1,
        cellsize=cellsize,
    )

    components = [
        Skycomponent(
            direction=phasecentre,
            frequency=frequency,
            polarisation_frame=image_pol,
            flux=flux,
        )
    ]

    blockvis = dft_skycomponent_visibility(blockvis, components)

    start = time.time()
    dirty = invert_ng(
        blockvis,
        model,
        normalise=True,
        verbosity=0,
        threads=args.threads,
    )[0]
    duration = time.time() - start
    assert numpy.abs(dirty["pixels"]).any() > 0.0, str(dirty)
    print(f"Scale {args.scale} threads {args.threads} duration {duration:.3f} s")


if __name__ == "__main__":
    main()
