"""Unit tests for visibility selectors

"""
import logging
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models import PolarisationFrame
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility.base import create_blockvisibility

log = logging.getLogger('rascil-logger')

log.setLevel(logging.WARNING)


class TestVisibilitySelectors(unittest.TestCase):
    def setUp(self):
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
        self.frequency = numpy.linspace(0.8e8, 1.2e8, 5)
        self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7, 1e7, 1e7])
        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])
        self.polarisation_frame = PolarisationFrame("linear")
        
        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
    
    def test_blockvisibility_groupby_time(self):
        bvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                      channel_bandwidth=self.channel_bandwidth,
                                      polarisation_frame=self.polarisation_frame,
                                      phasecentre=self.phasecentre, weight=1.0)
        times = numpy.array([result[0] for result in bvis.groupby("time")])
        assert times.all() == bvis.time.all()
    
    def test_blockvisibility_groupby_bins_time(self):
        bvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                      channel_bandwidth=self.channel_bandwidth,
                                      polarisation_frame=self.polarisation_frame,
                                      phasecentre=self.phasecentre, weight=1.0)
        for result in bvis.groupby_bins("time", 3):
            print(result[0])
    
    def test_blockvisibility_where_noflagged(self):
        bvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                      channel_bandwidth=self.channel_bandwidth,
                                      polarisation_frame=self.polarisation_frame,
                                      phasecentre=self.phasecentre, weight=1.0)
        selected_bvis = bvis.where(bvis["flags"] == 0)
        print(selected_bvis)
        print(bvis.blockvisibility_acc.size(), selected_bvis.blockvisibility_acc.size())
        assert len(selected_bvis.channel_bandwidth.shape) == 1
        assert len(selected_bvis.integration_time.shape) == 1

    def test_blockvisibility_where_absu(self):
        bvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                      channel_bandwidth=self.channel_bandwidth,
                                      polarisation_frame=self.polarisation_frame,
                                      phasecentre=self.phasecentre, weight=1.0)
        print(bvis)
        selected_bvis = bvis.where(numpy.abs(bvis.blockvisibility_acc.u_lambda) < 30.0)
        print(selected_bvis)
        print(bvis.blockvisibility_acc.size(), selected_bvis.blockvisibility_acc.size())
        assert len(selected_bvis.channel_bandwidth.shape) == 1
        assert len(selected_bvis.integration_time.shape) == 1

    def test_blockvisibility_select_time(self):
        bvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                      channel_bandwidth=self.channel_bandwidth,
                                      polarisation_frame=self.polarisation_frame,
                                      phasecentre=self.phasecentre, weight=1.0)
        selected_bvis = bvis.sel({"timeinx": slice(5, 7)})
        print(selected_bvis)
        assert len(selected_bvis.time) == 2
        assert len(selected_bvis.channel_bandwidth.shape) == 1
        assert len(selected_bvis.integration_time.shape) == 1

    def test_blockvisibility_where_frequency(self):
        bvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                      channel_bandwidth=self.channel_bandwidth,
                                      polarisation_frame=self.polarisation_frame,
                                      phasecentre=self.phasecentre, weight=1.0)
        cond = (bvis["frequency"] > 0.9e8) & (bvis["frequency"] < 1.2e8)
        selected_bvis = bvis.where(cond).dropna(dim="freqinx", how="all")
        print(selected_bvis)
        assert len(selected_bvis.frequency) == 2
        assert len(selected_bvis.channel_bandwidth.shape) == 1
        assert len(selected_bvis.integration_time.shape) == 1

    def test_blockvisibility_where_frequency_polarisation(self):
        bvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                      channel_bandwidth=self.channel_bandwidth,
                                      polarisation_frame=self.polarisation_frame,
                                      phasecentre=self.phasecentre, weight=1.0)
        cond = (bvis["frequency"] > 0.9e8) & (bvis["frequency"] < 1.2e8)
        selected_bvis = bvis.where(cond, drop=True).dropna(dim="freqinx", how="all")
        cond = (bvis["polarisation"] == "I") | (bvis["polarisation"] == "V")
        selected_bvis = selected_bvis.where(cond, drop=True).dropna(dim="polinx", how="all")
        print(selected_bvis)
        assert len(selected_bvis.frequency) == 2
        assert len(selected_bvis.channel_bandwidth.shape) == 1
        assert len(selected_bvis.integration_time.shape) == 1

    def test_blockvisibility_select_channel(self):
        bvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                      channel_bandwidth=self.channel_bandwidth,
                                      polarisation_frame=self.polarisation_frame,
                                      phasecentre=self.phasecentre, weight=1.0)
        selected_bvis = bvis.sel({"freqinx": slice(1, 3)})
        print(selected_bvis)
        assert len(selected_bvis.frequency) == 2
