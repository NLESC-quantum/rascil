""" Unit tests for RFI simulation

"""

import logging
import unittest
from unittest.mock import patch

import astropy.units as u
import numpy
import numpy.testing
from astropy.coordinates import SkyCoord

from rascil.data_models import PolarisationFrame, rascil_path
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.simulation.rfi import (
    calculate_rfi_at_station,
    calculate_station_correlation_rfi,
    simulate_rfi_block_prop,
    match_frequencies,
)
from rascil.processing_components import (
    create_blockvisibility,
    qa_image,
    export_image_to_fits,
    create_image_from_visibility,
    invert_ng,
)

log = logging.getLogger("rascil-logger")
log.setLevel(logging.WARNING)


class TestRFIRegression(unittest.TestCase):
    def setUp(self):
        pass

    def setup_telescope(self, telescope):
        """Initialise common elements"""
        # Set the random number so that we lways get the same answers
        numpy.random.seed(1805550721)

        self.nchannels = 5

        integration_time = 0.5
        self.ntimes = 100

        rmax = 100.0
        antskip = 1
        self.configuration = create_named_configuration(
            telescope, rmax=rmax, skip=antskip
        )
        self.nants = len(self.configuration.names)

        self.apparent_power = numpy.ones((self.ntimes, self.nants, self.nchannels))

        # Info. for dummy BlockVisibility
        ftimes = (numpy.pi / 43200.0) * numpy.arange(-1800.0, 1800.0, 180.0)
        if telescope == "MID":
            ffrequency = numpy.linspace(1.4e9, 1.9e9, 5)
            channel_bandwidth = numpy.array([1e8, 1e8, 1e8, 1e8, 1e8])
        else:
            ffrequency = numpy.linspace(1.3, 1.5e8, 5)
            channel_bandwidth = numpy.array([4e6, 4e6, 4e6, 4e6, 4e6])

        polarisation_frame = PolarisationFrame("linear")
        # Set the phasecentre so as cross the horizon since we don't have
        # the far sidelobes yet
        phasecentre = SkyCoord(
            ra=+180.0 * u.deg, dec=-62.8 * u.deg, frame="icrs", equinox="J2000"
        )
        self.bvis = create_blockvisibility(
            self.configuration,
            ftimes,
            ffrequency,
            channel_bandwidth=channel_bandwidth,
            polarisation_frame=polarisation_frame,
            phasecentre=phasecentre,
            weight=1.0,
        )

    def test_simulate_rfi_block_prop_image(self):
        """
        regression to test that simulate_rfi_block_prop correctly updates the
        block visibility data with RFI signal. We do this by making an image

        RFI signal is for the same frequency channels as the BlockVisibility has
        """
        self.setup_telescope("MID")
        nants_start = self.nants
        bvis = self.bvis.copy()

        emitter_power = numpy.zeros(
            (1, len(bvis.time), nants_start, len(bvis.frequency)), dtype=complex
        )  # one source
        # only add signal to the 4th and 5th channels (for testing purposes)
        emitter_power[:, :, :, 3] = 1.0e-10
        emitter_power[:, :, :, 4] = 5.0e-10
        emitter_coordinates = numpy.ones(
            (1, len(bvis.time), nants_start, 3),
        )
        # azimuth, elevation, distance
        emitter_coordinates[:, :, :, 0] = -170.0
        emitter_coordinates[:, :, :, 1] = 0.03
        emitter_coordinates[:, :, :, 2] = 600000.0

        simulate_rfi_block_prop(
            bvis,
            emitter_power,
            emitter_coordinates,
            ["source1"],
            bvis.frequency.values,
            beam_gain_state=None,
            use_pole=False,
        )

        model = create_image_from_visibility(bvis, npixel=2048, cellsize=0.001, nchan=1)
        dirty, sumwt = invert_ng(bvis, model)
        export_image_to_fits(dirty, rascil_path("test_results/test_rfi_image.fits"))
        qa = qa_image(dirty)
        print(qa)


if __name__ == "__main__":
    unittest.main()
