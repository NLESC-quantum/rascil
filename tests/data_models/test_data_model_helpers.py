""" Unit tests for data model helpers. The helpers facilitate persistence of data models
using HDF5


"""

import unittest
import logging

import astropy.units as u
import numpy
import xarray
from astropy.coordinates import SkyCoord

from rascil.data_models.data_model_helpers import (
    import_blockvisibility_from_hdf5,
    export_blockvisibility_to_hdf5,
    import_gaintable_from_hdf5,
    export_gaintable_to_hdf5,
    import_flagtable_from_hdf5,
    export_flagtable_to_hdf5,
    import_pointingtable_from_hdf5,
    export_pointingtable_to_hdf5,
    import_image_from_hdf5,
    export_image_to_hdf5,
    import_skycomponent_from_hdf5,
    export_skycomponent_to_hdf5,
    import_skymodel_from_hdf5,
    export_skymodel_to_hdf5,
    import_griddata_from_hdf5,
    export_griddata_to_hdf5,
    import_convolutionfunction_from_hdf5,
    export_convolutionfunction_to_hdf5,
)

from rascil.data_models.xarray_coordinate_support import image_wcs, griddata_wcs, cf_wcs
from rascil.data_models.memory_data_models import Skycomponent, SkyModel
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.image import create_image
from rascil.processing_components.calibration.operations import (
    create_gaintable_from_blockvisibility,
)
from rascil.processing_components.calibration.pointing import (
    create_pointingtable_from_blockvisibility,
)
from rascil.processing_components.imaging import dft_skycomponent_visibility
from rascil.processing_components.simulation import simulate_gaintable
from rascil.processing_components.simulation.pointing import simulate_pointingtable
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility.base import create_blockvisibility
from rascil.processing_components.flagging.base import (
    create_flagtable_from_blockvisibility,
)
from rascil.processing_components.griddata.operations import create_griddata_from_image
from rascil.processing_components.griddata import create_convolutionfunction_from_image

log = logging.getLogger("rascil-logger")

log.setLevel(logging.INFO)


class TestDataModelHelpers(unittest.TestCase):
    def _data_model_equals(self, ds_new, ds_ref):
        """Check if two xarray objects are identical except to values

        Precision in lost in HDF files at close to the machine precision so we cannot
        reliably use xarray.equals(). So this function is specific to this set of tests

        Throws AssertionError or returns True

        :param ds_ref: xarray Dataset or DataArray
        :param ds_new: xarray Dataset or DataArray
        :return: True or False
        """
        for coord in ds_ref.coords:
            assert coord in ds_new.coords
        for coord in ds_new.coords:
            assert coord in ds_ref.coords
        for var in ds_ref.data_vars:
            assert var in ds_new.data_vars
        for var in ds_new.data_vars:
            assert var in ds_ref.data_vars
        for attr in ds_ref.attrs.keys():
            assert attr in ds_new.attrs.keys()
        for attr in ds_new.attrs.keys():
            assert attr in ds_ref.attrs.keys()

        return True

    def setUp(self):
        from rascil.data_models.parameters import rascil_path, rascil_data_path

        self.results_dir = rascil_path("test_results")

        self.mid = create_named_configuration("MID", rmax=1000.0)
        self.times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 100.0)
        self.frequency = numpy.linspace(1.0e8, 1.1e8, 3)
        self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7])
        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])

        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(
            ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.compabsdirection = SkyCoord(
            ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.comp = Skycomponent(
            direction=self.compabsdirection, frequency=self.frequency, flux=self.flux
        )

        self.verbose = False

    def test_readwriteblockvisibility(self):
        self.vis = create_blockvisibility(
            self.mid,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            phasecentre=self.phasecentre,
            polarisation_frame=PolarisationFrame("linear"),
            weight=1.0,
        )
        self.vis = dft_skycomponent_visibility(self.vis, self.comp)
        if self.verbose:
            print(self.vis)
            print(self.vis.configuration)
        export_blockvisibility_to_hdf5(
            self.vis,
            "%s/test_data_model_helpers_blockvisibility.hdf" % self.results_dir,
        )
        newvis = import_blockvisibility_from_hdf5(
            "%s/test_data_model_helpers_blockvisibility.hdf" % self.results_dir
        )
        assert self._data_model_equals(newvis, self.vis)

    def test_readwritegaintable(self):
        self.vis = create_blockvisibility(
            self.mid,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            phasecentre=self.phasecentre,
            polarisation_frame=PolarisationFrame("linear"),
            weight=1.0,
        )
        gt = create_gaintable_from_blockvisibility(self.vis, timeslice="auto")
        gt = simulate_gaintable(gt, phase_error=1.0, amplitude_error=0.1)
        if self.verbose:
            print(gt)
        export_gaintable_to_hdf5(
            gt, "%s/test_data_model_helpers_gaintable.hdf" % self.results_dir
        )
        newgt = import_gaintable_from_hdf5(
            "%s/test_data_model_helpers_gaintable.hdf" % self.results_dir
        )
        assert self._data_model_equals(newgt, gt)

    def test_readwriteflagtable(self):
        self.vis = create_blockvisibility(
            self.mid,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            phasecentre=self.phasecentre,
            polarisation_frame=PolarisationFrame("linear"),
            weight=1.0,
        )
        ft = create_flagtable_from_blockvisibility(self.vis, timeslice="auto")
        if self.verbose:
            print(ft)
        export_flagtable_to_hdf5(
            ft, "%s/test_data_model_helpers_flagtable.hdf" % self.results_dir
        )
        newft = import_flagtable_from_hdf5(
            "%s/test_data_model_helpers_flagtable.hdf" % self.results_dir
        )
        assert self._data_model_equals(newft, ft)

    def test_readwritepointingtable(self):
        self.vis = create_blockvisibility(
            self.mid,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            phasecentre=self.phasecentre,
            polarisation_frame=PolarisationFrame("linear"),
            weight=1.0,
        )
        pt = create_pointingtable_from_blockvisibility(self.vis, timeslice="auto")
        pt = simulate_pointingtable(pt, pointing_error=0.001)
        if self.verbose:
            print(pt)
        export_pointingtable_to_hdf5(
            pt, "%s/test_data_model_helpers_pointingtable.hdf" % self.results_dir
        )
        newpt = import_pointingtable_from_hdf5(
            "%s/test_data_model_helpers_pointingtable.hdf" % self.results_dir
        )
        assert self._data_model_equals(newpt, pt)

    def test_readwriteimage(self):
        im = create_image(
            phasecentre=self.phasecentre,
            frequency=self.frequency,
            npixel=256,
            polarisation_frame=PolarisationFrame("stokesIQUV"),
        )
        im["pixels"][...] = 1.0
        export_image_to_hdf5(
            im, "%s/test_data_model_helpers_image.hdf" % self.results_dir
        )
        newim = import_image_from_hdf5(
            "%s/test_data_model_helpers_image.hdf" % self.results_dir
        )
        assert self._data_model_equals(newim, im)

    def test_readwriteimage_zarr(self):
        """Test to see if an image can be written to and read from a zarr file"""
        im = create_image(
            phasecentre=self.phasecentre,
            frequency=self.frequency,
            npixel=256,
            polarisation_frame=PolarisationFrame("stokesIQUV"),
        )
        rand = numpy.random.random(im["pixels"].shape)
        im["pixels"][...] = rand
        if self.verbose:
            print(im)
        import os

        # We cannot save dicts to a netcdf file
        im.attrs["clean_beam"] = ""

        store = os.path.expanduser(
            "%s/test_data_model_helpers_image.zarr" % self.results_dir
        )
        im.to_zarr(
            store=store,
            chunk_store=store,
            mode="w",
        )
        del im
        newim = xarray.open_zarr(store, chunk_store=store)
        assert newim["pixels"].data.compute().all() == rand.all()

    def test_readwriteskycomponent(self):
        export_skycomponent_to_hdf5(
            self.comp, "%s/test_data_model_helpers_skycomponent.hdf" % self.results_dir
        )
        newsc = import_skycomponent_from_hdf5(
            "%s/test_data_model_helpers_skycomponent.hdf" % self.results_dir
        )

        assert newsc.flux.shape == self.comp.flux.shape
        assert numpy.max(numpy.abs(newsc.flux - self.comp.flux)) < 1e-15

    def test_readwriteskymodel(self):
        self.vis = create_blockvisibility(
            self.mid,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            phasecentre=self.phasecentre,
            polarisation_frame=PolarisationFrame("linear"),
            weight=1.0,
        )
        im = create_image(
            phasecentre=self.phasecentre,
            frequency=self.frequency,
            npixel=256,
            polarisation_frame=PolarisationFrame("stokesIQUV"),
        )
        gt = create_gaintable_from_blockvisibility(self.vis, timeslice="auto")
        sm = SkyModel(components=[self.comp], image=im, gaintable=gt)
        export_skymodel_to_hdf5(
            sm, "%s/test_data_model_helpers_skymodel.hdf" % self.results_dir
        )
        newsm = import_skymodel_from_hdf5(
            "%s/test_data_model_helpers_skymodel.hdf" % self.results_dir
        )

        assert newsm.components[0].flux.shape == self.comp.flux.shape

    def test_readwritegriddata(self):
        # This fails on comparison of the v axis.
        im = create_image(
            phasecentre=self.phasecentre,
            frequency=self.frequency,
            npixel=256,
            polarisation_frame=PolarisationFrame("stokesIQUV"),
        )
        gd = create_griddata_from_image(im)
        export_griddata_to_hdf5(
            gd, "%s/test_data_model_helpers_griddata.hdf" % self.results_dir
        )
        newgd = import_griddata_from_hdf5(
            "%s/test_data_model_helpers_griddata.hdf" % self.results_dir
        )
        assert self._data_model_equals(newgd, gd)

    def test_readwriteconvolutionfunction(self):
        # This fails on comparison of the v axis.
        im = create_image(
            phasecentre=self.phasecentre,
            frequency=self.frequency,
            npixel=256,
            polarisation_frame=PolarisationFrame("stokesIQUV"),
        )
        cf = create_convolutionfunction_from_image(im)
        if self.verbose:
            print(cf)
        export_convolutionfunction_to_hdf5(
            cf, "%s/test_data_model_helpers_convolutionfunction.hdf" % self.results_dir
        )
        newcf = import_convolutionfunction_from_hdf5(
            "%s/test_data_model_helpers_convolutionfunction.hdf" % self.results_dir
        )

        assert self._data_model_equals(newcf, cf)


if __name__ == "__main__":
    unittest.main()
