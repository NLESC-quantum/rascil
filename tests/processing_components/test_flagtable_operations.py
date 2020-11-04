""" Unit tests for visibility operations


"""

import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models.memory_data_models import Skycomponent
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import create_flagtable_from_blockvisibility, \
    create_blockvisibility
from rascil.processing_components.simulation import create_named_configuration


class TestFlagTableOperations(unittest.TestCase):
    def setUp(self):
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
        self.frequency = numpy.linspace(1.0e8, 1.1e8, 3)
        self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7])
        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])
        self.polarisation_frame = PolarisationFrame("linear")
        
        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.compabsdirection = SkyCoord(ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        pcof = self.phasecentre.skyoffset_frame()
        self.compreldirection = self.compabsdirection.transform_to(pcof)
        self.comp = Skycomponent(direction=self.compreldirection, frequency=self.frequency, flux=self.flux)
    
    def test_create_flagtable(self):
        bvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                      channel_bandwidth=self.channel_bandwidth,
                                      phasecentre=self.phasecentre,
                                      polarisation_frame=self.polarisation_frame,
                                      weight=1.0)
        ft = create_flagtable_from_blockvisibility(bvis)
        assert ft.flags.shape == bvis.vis.shape

    def test_flagtable_groupby_time(self):
        bvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                      channel_bandwidth=self.channel_bandwidth,
                                      polarisation_frame=self.polarisation_frame,
                                      phasecentre=self.phasecentre, weight=1.0)
        ft = create_flagtable_from_blockvisibility(bvis)
        times = numpy.array([result[0] for result in ft.groupby("time")])
        assert times.all() == bvis.time.all()

    def test_flagtable_groupby_bins_time(self):
        bvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                      channel_bandwidth=self.channel_bandwidth,
                                      polarisation_frame=self.polarisation_frame,
                                      phasecentre=self.phasecentre, weight=1.0)
        ft = create_flagtable_from_blockvisibility(bvis)
        for result in ft.groupby_bins("time", 3):
            print(result[0])

    def test_flagtable_where(self):
        bvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                      channel_bandwidth=self.channel_bandwidth,
                                      polarisation_frame=self.polarisation_frame,
                                      phasecentre=self.phasecentre, weight=1.0)
        ft = create_flagtable_from_blockvisibility(bvis)
        print(ft.where(ft["flags"]==0))

    def test_flagtable_select_time(self):
        bvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                      channel_bandwidth=self.channel_bandwidth,
                                      polarisation_frame=self.polarisation_frame,
                                      phasecentre=self.phasecentre, weight=1.0)
        ft = create_flagtable_from_blockvisibility(bvis)
        times = ft.time
        selected_ft = ft.sel({"time": slice(times[1], times[2])})
        assert len(selected_ft.time) == 2

    def test_flagtable_select_frequency(self):
        bvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                      channel_bandwidth=self.channel_bandwidth,
                                      polarisation_frame=self.polarisation_frame,
                                      phasecentre=self.phasecentre, weight=1.0)
        ft = create_flagtable_from_blockvisibility(bvis)
        frequency = ft.frequency
        selected_ft = ft.sel({"frequency": slice(frequency[1], frequency[2])})
        assert len(selected_ft.frequency) == 2


if __name__ == '__main__':
    unittest.main()
