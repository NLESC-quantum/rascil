""" Unit tests for visibility operations


"""

import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models.memory_data_models import Skycomponent
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import create_blockvisibility
from rascil.processing_components.flagging.operations import flagging_blockvisibility
from rascil.processing_components.simulation import create_named_configuration


class TestFlaggingOperations(unittest.TestCase):
    def setUp(self):
        self.lowcore = create_named_configuration("LOWBD2-CORE")
        self.times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
        self.frequency = numpy.linspace(1.0e8, 1.1e8, 3)
        self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7])
        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])
        self.polarisation_frame = PolarisationFrame("linear")

        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(
            ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.compabsdirection = SkyCoord(
            ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
        )
        pcof = self.phasecentre.skyoffset_frame()
        self.compreldirection = self.compabsdirection.transform_to(pcof)
        self.comp = Skycomponent(
            direction=self.compreldirection, frequency=self.frequency, flux=self.flux
        )

    def test_flagging_blockvisibility_multiple(self):
        bvis = create_blockvisibility(
            self.lowcore,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            phasecentre=self.phasecentre,
            polarisation_frame=self.polarisation_frame,
            weight=1.0,
        )
        antennas = [1, 3]
        channels = [0, 1]
        pols = [0]
        bvis = flagging_blockvisibility(
            bvis, antennas=antennas, channels=channels, polarisations=pols
        )
        # Check flagging on channels
        for channel in channels:
            assert bvis["flags"].data[..., channel, :].all() == 1
        # Check flagging on pols
        for pol in pols:
            assert (bvis["flags"].data[..., pol] == 1).all()
        # Check flagging on antennas
        for ibaseline, (a1, a2) in enumerate(bvis.baselines.data):
            if a1 in antennas or a2 in antennas:
                assert (bvis["flags"].data[:, ibaseline, ...] == 1).all()


if __name__ == "__main__":
    unittest.main()
