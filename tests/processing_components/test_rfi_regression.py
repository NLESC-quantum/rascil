""" Unit tests for RFI simulation

"""

import os
import logging
import unittest

import astropy.units as u
import numpy
import numpy.testing
from astropy.coordinates import SkyCoord

from rascil.data_models import PolarisationFrame, rascil_path
from rascil.processing_components import (
    create_blockvisibility,
    qa_image,
    export_image_to_fits,
    create_image_from_visibility,
    invert_ng,
)
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.simulation.rfi import (
    simulate_rfi_block_prop,
)

log = logging.getLogger("rascil-logger")
log.setLevel(logging.WARNING)


class TestRFIRegression(unittest.TestCase):
    def setUp(self):
        self.persist = os.getenv("RASCIL_PERSIST", False)

    def setup_telescope(self, telescope):
        """Initialise common elements"""
        # Set the random number so that we lways get the same answers
        numpy.random.seed(1805550721)

        self.nchannels = 5

        self.ntimes = 100

        rmax = 50.0
        antskip = 1
        self.configuration = create_named_configuration(
            telescope, rmax=rmax, skip=antskip
        )
        self.nants = len(self.configuration.names)

        self.apparent_power = numpy.ones((self.ntimes, self.nants, self.nchannels))

        ftimes = (numpy.pi / 43200.0) * numpy.arange(-3600, +3600.0, 225.0)
        log.info(f"Times: {ftimes}")
        if telescope == "MID":
            ffrequency = numpy.linspace(1.4e9, 1.9e9, 5)
            channel_bandwidth = numpy.array([1e8, 1e8, 1e8, 1e8, 1e8])
        else:
            ffrequency = numpy.linspace(1.3, 1.5e8, 5)
            channel_bandwidth = numpy.array([4e6, 4e6, 4e6, 4e6, 4e6])

        polarisation_frame = PolarisationFrame("linear")
        # Set the phasecentre close to the horizon since we don't have
        # the far sidelobes yet
        phasecentre = SkyCoord(
            ra=0.0 * u.deg, dec=+30.0 * u.deg, frame="icrs", equinox="J2000"
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
        # azimuth, elevation, distance
        emitter_coordinates[:, :, :, 0] = (
            0.0 + numpy.linspace(-1.0, 1.0, 5)[numpy.newaxis, numpy.newaxis, :]
        )
        emitter_coordinates[:, :, :, 1] = (
            29.45 + numpy.linspace(-2.0, 2.0, 5)[numpy.newaxis, numpy.newaxis, :]
        )
        emitter_coordinates[:, :, :, 2] = (
            600000.0
            + numpy.linspace(-5000.0, 5000.0, 5)[numpy.newaxis, numpy.newaxis, :]
        )
        for apply_primary_beam in [True, False]:
            simulate_rfi_block_prop(
                bvis,
                emitter_power,
                emitter_coordinates,
                ["source1"],
                bvis.frequency.values,
                beam_gain_state=None,
                use_pole=False,
                apply_primary_beam=apply_primary_beam,
            )

            model = create_image_from_visibility(
                bvis, npixel=1024, cellsize=0.002, nchan=1
            )
            dirty, sumwt = invert_ng(bvis, model)
            dirty["pixels"].data[numpy.abs(dirty["pixels"].data) > 1e6] = 0.0
            if self.persist:
                export_image_to_fits(
                    dirty,
                    rascil_path(
                        f"test_results/test_rfi_image_withPB{apply_primary_beam}.fits"
                    ),
                )
            qa = qa_image(dirty)

            if apply_primary_beam:
                numpy.testing.assert_almost_equal(
                    qa.data["max"], 458854.2087080175, err_msg=str(qa)
                )
                numpy.testing.assert_almost_equal(
                    qa.data["min"], -317039.1555572104, err_msg=str(qa)
                )
            else:
                numpy.testing.assert_almost_equal(
                    qa.data["max"], 999962.8165061118, err_msg=str(qa)
                )
                numpy.testing.assert_almost_equal(
                    qa.data["min"], -999962.870884174, err_msg=str(qa)
                )


if __name__ == "__main__":
    unittest.main()
