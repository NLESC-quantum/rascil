""" Unit tests for pipelines expressed via rsexecute
"""

import functools
import logging
import os
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import (
    export_image_to_fits,
    smooth_image,
)
from rascil.processing_components.griddata.kernels import (
    create_awterm_convolutionfunction,
)
from rascil.processing_components.imaging import dft_skycomponent_visibility
from rascil.processing_components.simulation import (
    create_named_configuration,
    decimate_configuration,
)
from rascil.processing_components.simulation import (
    ingest_unittest_visibility,
    create_unittest_model,
    insert_unittest_errors,
    create_unittest_components,
)
from rascil.processing_components.skycomponent.operations import (
    find_skycomponents,
    find_nearest_skycomponent,
    insert_skycomponent,
)
from rascil.workflows.rsexecute.execution_support.rsexecute import (
    rsexecute,
    get_dask_client,
)
from rascil.workflows.rsexecute.imaging.imaging_rsexecute import (
    zero_list_rsexecute_workflow,
    invert_list_rsexecute_workflow,
    subtract_list_rsexecute_workflow,
    predict_list_rsexecute_workflow,
)
from rascil.processing_components.imaging.imaging_helpers import sum_invert_results

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))


class TestImaging(unittest.TestCase):
    def setUp(self):

        # To make WAGG GPU-based module working in Dask it is required to use only one thread per worker,
        # to avoid the hardware resource race by the different threads which causes cudaError (invalid CUDA context).
        rsexecute.set_client(
            client=get_dask_client(n_workers=4, threads_per_worker=1),
            use_dask=True,
            verbose=True,
        )

        from rascil.data_models.parameters import rascil_path

        self.test_dir = rascil_path("test_results")

        self.persist = os.getenv("RASCIL_PERSIST", False)

    def tearDown(self):
        rsexecute.close()

    def actualSetUp(
        self,
        add_errors=False,
        freqwin=3,
        dospectral=True,
        dopol=False,
        zerow=False,
    ):

        self.npixel = 256
        self.low = create_named_configuration("LOWBD2", rmax=750.0)
        self.low = decimate_configuration(self.low, skip=6)
        self.freqwin = freqwin
        self.bvis_list = list()
        self.ntimes = 5
        self.cellsize = 0.0005
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

        if dospectral:
            flux = numpy.array(
                [f * numpy.power(freq / 1e8, -0.7) for freq in self.frequency]
            )
        else:
            flux = numpy.array([f])

        self.phasecentre = SkyCoord(
            ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.bvis_list = [
            rsexecute.execute(ingest_unittest_visibility)(
                self.low,
                numpy.array([self.frequency[freqwin]]),
                numpy.array([self.channelwidth[freqwin]]),
                self.times,
                self.vis_pol,
                self.phasecentre,
                zerow=zerow,
            )
            for freqwin, _ in enumerate(self.frequency)
        ]

        self.model_list = [
            rsexecute.execute(create_unittest_model, nout=freqwin)(
                self.bvis_list[freqwin],
                self.image_pol,
                cellsize=self.cellsize,
                npixel=self.npixel,
            )
            for freqwin, _ in enumerate(self.frequency)
        ]

        self.components_list = [
            rsexecute.execute(create_unittest_components)(
                self.model_list[freqwin],
                flux[freqwin, :][numpy.newaxis, :],
                single=False,
            )
            for freqwin, _ in enumerate(self.frequency)
        ]

        self.components_list = rsexecute.compute(self.components_list, sync=True)

        self.model_list = [
            rsexecute.execute(insert_skycomponent, nout=1)(
                self.model_list[freqwin], self.components_list[freqwin]
            )
            for freqwin, _ in enumerate(self.frequency)
        ]

        self.model_list = rsexecute.compute(self.model_list, sync=True)

        self.bvis_list = [
            rsexecute.execute(dft_skycomponent_visibility)(
                self.bvis_list[freqwin], self.components_list[freqwin]
            )
            for freqwin, _ in enumerate(self.frequency)
        ]
        centre = self.freqwin // 2
        # Calculate the model convolved with a Gaussian.
        self.model = self.model_list[centre]

        self.cmodel = smooth_image(self.model)
        if self.persist:
            export_image_to_fits(
                self.model, "%s/test_imaging_model.fits" % self.test_dir
            )
            export_image_to_fits(
                self.cmodel, "%s/test_imaging_cmodel.fits" % self.test_dir
            )

        if add_errors:
            self.bvis_list = [
                rsexecute.execute(insert_unittest_errors)(self.bvis_list[i])
                for i, _ in enumerate(self.frequency)
            ]

        self.components = self.components_list[centre]

        self.gcfcf = functools.partial(
            create_awterm_convolutionfunction,
            nw=50,
            wstep=16.0,
            oversampling=4,
            support=100,
            use_aaf=True,
            polarisation_frame=self.vis_pol,
        )

    def _check_components(self, dirty, fluxthreshold=0.6, positionthreshold=1.0):
        comps = find_skycomponents(
            dirty, fwhm=1.0, threshold=10 * fluxthreshold, npixels=5
        )
        assert len(comps) == len(
            self.components
        ), "Different number of components found: original %d, recovered %d" % (
            len(self.components),
            len(comps),
        )
        cellsize = abs(dirty.image_acc.wcs.wcs.cdelt[0])

        for comp in comps:
            # Check for agreement in direction
            ocomp, separation = find_nearest_skycomponent(
                comp.direction, self.components
            )
            assert separation / cellsize < positionthreshold, (
                "Component differs in position %.3f pixels" % separation / cellsize
            )

    def _predict_base(
        self,
        context="wg",
        do_wstacking=True,
        extra="",
        fluxthreshold=1.0,
        normalise=True,
        **kwargs
    ):
        centre = self.freqwin // 2

        vis_list = zero_list_rsexecute_workflow(self.bvis_list)
        vis_list = predict_list_rsexecute_workflow(
            vis_list, self.model_list, context=context, **kwargs
        )
        vis_list = subtract_list_rsexecute_workflow(self.bvis_list, vis_list)
        vis_list = rsexecute.compute(vis_list, sync=True)

        dirty = invert_list_rsexecute_workflow(
            vis_list,
            self.model_list,
            context=context,
            dopsf=False,
            normalise=normalise,
            **kwargs
        )
        dirty = rsexecute.compute(dirty, sync=True)[centre]

        assert numpy.max(numpy.abs(dirty[0]["pixels"].data)), "Residual image is empty"
        if self.persist:
            export_image_to_fits(
                dirty[0],
                "%s/test_imaging_predict_%s%s_%s_dirty.fits"
                % (self.test_dir, context, extra, rsexecute.type()),
            )

        maxabs = numpy.max(numpy.abs(dirty[0]["pixels"].data))
        assert maxabs < fluxthreshold, "Error %.3f greater than fluxthreshold %.3f " % (
            maxabs,
            fluxthreshold,
        )

    def _invert_base(
        self,
        context,
        extra="",
        fluxthreshold=2.0,
        positionthreshold=1.0,
        check_components=True,
        normalise=True,
        gcfcf=None,
        dopsf=False,
        **kwargs
    ):

        centre = self.freqwin // 2
        dirty = invert_list_rsexecute_workflow(
            self.bvis_list,
            self.model_list,
            context=context,
            dopsf=dopsf,
            normalise=normalise,
            gcfcf=gcfcf,
            **kwargs
        )
        dirty = rsexecute.compute(dirty, sync=True)[centre]

        if self.persist:
            if dopsf:
                export_image_to_fits(
                    dirty[0],
                    "%s/test_imaging_invert_%s%s_%s_psf.fits"
                    % (self.test_dir, context, extra, rsexecute.type()),
                )
            else:
                export_image_to_fits(
                    dirty[0],
                    "%s/test_imaging_invert_%s%s_%s_dirty.fits"
                    % (self.test_dir, context, extra, rsexecute.type()),
                )

        assert numpy.max(numpy.abs(dirty[0]["pixels"].data)), "Image is empty"

        if check_components:
            self._check_components(dirty[0], fluxthreshold, positionthreshold)

    def test_predict_wg(self):
        self.actualSetUp()
        self._predict_base(context="wg", fluxthreshold=0.62)

    def test_invert_wg(self):
        self.actualSetUp(add_errors=False)
        self._invert_base(context="wg", positionthreshold=2.0, check_components=True)


if __name__ == "__main__":
    unittest.main()
