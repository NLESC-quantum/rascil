""" Unit tests for pipelines expressed via rsexecute
"""

import logging
import os
import sys
import unittest
import copy

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import create_named_configuration
from rascil.processing_components import (
    ingest_unittest_visibility,
    create_low_test_skymodel_from_gleam,
    create_pb,
    qa_image
)
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute
from rascil.workflows.rsexecute.skymodel.skymodel_rsexecute import (
    predict_skymodel_list_rsexecute_workflow,
    sum_skymodels_rsexecute
)

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))


class TestSkyModel(unittest.TestCase):
    def setUp(self):

        rsexecute.set_client(use_dask=True)

        from rascil.data_models.parameters import rascil_path

        self.dir = rascil_path("test_results")

        self.persist = os.getenv("RASCIL_PERSIST", False)

    def tearDown(self):
        rsexecute.close()

    def actualSetUp(self, freqwin=1, dopol=False, zerow=False):

        self.npixel = 512
        self.low = create_named_configuration("LOWBD2", rmax=300.0)
        self.freqwin = freqwin
        self.vis_list = list()
        self.ntimes = 5
        self.cellsize = 0.001
        self.radius = self.npixel * self.cellsize / 2.0
        # Choose the interval so that the maximum change in w is smallish
        integration_time = numpy.pi * (24 / (12 * 60))
        self.times = numpy.linspace(
            -integration_time * (self.ntimes // 2),
            integration_time * (self.ntimes // 2),
            self.ntimes,
        )

        if freqwin > 1:
            self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
            self.channelwidth = numpy.array(
                freqwin * [self.frequency[1] - self.frequency[0]]
            )
        else:
            self.frequency = numpy.array([1.0e8])
            self.channelwidth = numpy.array([4e7])

        if dopol:
            self.vis_pol = PolarisationFrame("linear")
            self.image_pol = PolarisationFrame("stokesIQUV")
            f = numpy.array([100.0, 20.0, -10.0, 1.0])
        else:
            self.vis_pol = PolarisationFrame("stokesI")
            self.image_pol = PolarisationFrame("stokesI")
            f = numpy.array([100.0])

        self.phasecentre = SkyCoord(
            ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.vis_list = [
            rsexecute.execute(ingest_unittest_visibility)(
                self.low,
                [self.frequency[freqwin]],
                [self.channelwidth[freqwin]],
                self.times,
                self.vis_pol,
                self.phasecentre,
                zerow=zerow,
            )
            for freqwin, _ in enumerate(self.frequency)
        ]
        self.vis_list = rsexecute.compute(self.vis_list)

    def test_predict(self):
        self.actualSetUp()

        self.skymodel_list = [
            rsexecute.execute(create_low_test_skymodel_from_gleam)(
                npixel=self.npixel,
                cellsize=self.cellsize,
                frequency=[self.frequency[f]],
                phasecentre=self.phasecentre,
                radius=self.radius,
                polarisation_frame=PolarisationFrame("stokesI"),
                flux_limit=0.3,
                flux_threshold=1.0,
                flux_max=5.0,
            )
            for f, freq in enumerate(self.frequency)
        ]

        self.skymodel_list = rsexecute.compute(self.skymodel_list, sync=True)

        assert len(self.skymodel_list[0].components) == 11, len(
            self.skymodel_list[0].components
        )
        assert (
            numpy.max(numpy.abs(self.skymodel_list[0].image["pixels"].data)) > 0.0
        ), "Image is empty"

        self.skymodel_list = rsexecute.scatter(self.skymodel_list)
        skymodel_vislist = predict_skymodel_list_rsexecute_workflow(
            self.vis_list[0], self.skymodel_list, context="ng"
        )
        skymodel_vislist = rsexecute.compute(skymodel_vislist, sync=True)
        assert numpy.max(numpy.abs(skymodel_vislist[0].vis)) > 0.0

    def test_predict_nocomponents(self):
        self.actualSetUp()

        self.skymodel_list = [
            rsexecute.execute(create_low_test_skymodel_from_gleam)(
                npixel=self.npixel,
                cellsize=self.cellsize,
                frequency=[self.frequency[f]],
                radius=self.radius,
                phasecentre=self.phasecentre,
                polarisation_frame=PolarisationFrame("stokesI"),
                flux_limit=0.3,
                flux_threshold=1.0,
                flux_max=5.0,
            )
            for f, freq in enumerate(self.frequency)
        ]

        self.skymodel_list = rsexecute.compute(self.skymodel_list, sync=True)

        for i, sm in enumerate(self.skymodel_list):
            sm.components = []

        assert (
            numpy.max(numpy.abs(self.skymodel_list[0].image["pixels"].data)) > 0.0
        ), "Image is empty"

        self.skymodel_list = rsexecute.scatter(self.skymodel_list)
        skymodel_vislist = predict_skymodel_list_rsexecute_workflow(
            self.vis_list[0], self.skymodel_list, context="ng"
        )
        skymodel_vislist = rsexecute.compute(skymodel_vislist, sync=True)
        assert numpy.max(numpy.abs(skymodel_vislist[0].vis)) > 0.0

    def test_predict_noimage(self):
        self.actualSetUp()

        self.skymodel_list = [
            rsexecute.execute(create_low_test_skymodel_from_gleam)(
                npixel=self.npixel,
                cellsize=self.cellsize,
                frequency=[self.frequency[f]],
                radius=self.radius,
                phasecentre=self.phasecentre,
                polarisation_frame=PolarisationFrame("stokesI"),
                flux_limit=0.3,
                flux_threshold=1.0,
                flux_max=5.0,
            )
            for f, freq in enumerate(self.frequency)
        ]

        self.skymodel_list = rsexecute.compute(self.skymodel_list, sync=True)
        for i, sm in enumerate(self.skymodel_list):
            sm.image = None

        ##assert isinstance(self.skymodel_list[0].components[0], Skycomponent), self.skymodel_list[0].components[0]
        assert len(self.skymodel_list[0].components) == 11, len(
            self.skymodel_list[0].components
        )

        self.skymodel_list = rsexecute.scatter(self.skymodel_list)
        skymodel_vislist = predict_skymodel_list_rsexecute_workflow(
            self.vis_list[0], self.skymodel_list, context="ng"
        )
        skymodel_vislist = rsexecute.compute(skymodel_vislist, sync=True)
        assert numpy.max(numpy.abs(skymodel_vislist[0].vis)) > 0.0

    def test_sum_skymodels(self):
        self.actualSetUp()

        self.skymodel_list = [
            rsexecute.execute(create_low_test_skymodel_from_gleam)(
                npixel=self.npixel,
                cellsize=self.cellsize,
                frequency=[self.frequency[f]],
                radius=self.radius,
                phasecentre=self.phasecentre,
                polarisation_frame=PolarisationFrame("stokesI"),
                flux_limit=0.3,
                flux_threshold=1.0,
                flux_max=5.0,
            )
            for f, freq in enumerate(self.frequency)
        ]
        def skymodel_set_pb(sm):
            sm.mask = create_pb(sm.image, "LOW")
            return sm
            
        self.skymodel_list = [rsexecute.execute(skymodel_set_pb)(sm)
                              for sm in self.skymodel_list]

        sum_skymodel_list = sum_skymodels_rsexecute(self.skymodel_list)
        sum_skymodel = rsexecute.compute(sum_skymodel_list, sync=True)
        qa = qa_image(sum_skymodel.image)
        numpy.testing.assert_allclose(qa.data["max"], 4.959490911894567, atol=1e-7, err_msg=f"{qa}")
        numpy.testing.assert_allclose(qa.data["min"], 0.0, atol=1e-7, err_msg=f"{qa}")
        qa = qa_image(sum_skymodel.mask)
        numpy.testing.assert_allclose(qa.data["max"], 0.9999999999999988, atol=1e-7, err_msg=f"{qa}")
        numpy.testing.assert_allclose(qa.data["min"], 1.716481246836587e-06, atol=1e-7, err_msg=f"{qa}")


if __name__ == "__main__":
    unittest.main()
