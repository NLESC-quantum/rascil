"""Unit tests for primary beam application with polarisation


"""

import logging
import os
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.testing import assert_array_almost_equal

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import (
    export_image_to_fits,
    reproject_image,
    apply_voltage_pattern_to_image,
    qa_image,
    create_image_from_array,
    invert_blockvisibility,
)
from rascil.processing_components.imaging import (
    create_image_from_visibility,
)
from rascil.processing_components.imaging.dft import (
    dft_skycomponent_visibility,
    idft_visibility_skycomponent,
)
from rascil.processing_components.imaging.primary_beams import create_vp, create_pb
from rascil.processing_components.simulation import (
    create_named_configuration,
    decimate_configuration,
)
from rascil.processing_components.skycomponent import (
    create_skycomponent,
    apply_beam_to_skycomponent,
    copy_skycomponent,
    apply_voltage_pattern_to_skycomponent,
    find_skycomponents,
)
from rascil.processing_components.visibility import create_blockvisibility

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)


class TestPrimaryBeamsPol(unittest.TestCase):
    def setUp(self):
        from rascil.data_models.parameters import rascil_path

        self.results_dir = rascil_path("test_results")

        self.persist = os.getenv("RASCIL_PERSIST", False)

    def createVis(self, config="MID", dec=-35.0, rmax=1e3, freq=1.3e9):
        self.frequency = numpy.array([freq])
        self.channel_bandwidth = numpy.array([1e6])
        self.flux = numpy.array([[100.0, 60.0, -10.0, +1.0]])
        self.phasecentre = SkyCoord(
            ra=+15.0 * u.deg, dec=dec * u.deg, frame="icrs", equinox="J2000"
        )
        self.config = create_named_configuration(config, rmax=rmax)
        self.times = numpy.linspace(-300.0, 300.0, 3) * numpy.pi / 43200.0
        nants = self.config.xyz.shape[0]
        self.npixel = 256
        self.fov = 8
        self.cellsize = numpy.pi * self.fov / (self.npixel * 180.0)
        assert nants > 1
        assert len(self.config.names) == nants
        assert len(self.config.mount) == nants

    def test_apply_primary_beam_imageplane(self):
        self.createVis()
        telescope = "MID"
        lflux = numpy.array([[100.0, 1.0, -10.0, +60.0]])
        cflux = numpy.array([[100.0, 60.0, -10.0, +1.0]])
        apply_pb = True
        for flux, vpol in (
            (lflux, PolarisationFrame("linear")),
            (cflux, PolarisationFrame("circular")),
        ):
            # print("Testing {0}".format(vpol.type))
            # print("Original flux = {}".format(flux))
            bvis = create_blockvisibility(
                self.config,
                self.times,
                self.frequency,
                channel_bandwidth=self.channel_bandwidth,
                phasecentre=self.phasecentre,
                weight=1.0,
                polarisation_frame=vpol,
                zerow=True,
            )

            component_centre = SkyCoord(
                ra=+15.5 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
            )
            component = create_skycomponent(
                direction=component_centre,
                flux=flux,
                frequency=self.frequency,
                polarisation_frame=PolarisationFrame("stokesIQUV"),
            )
            model = create_image_from_visibility(
                bvis,
                cellsize=self.cellsize,
                npixel=self.npixel,
                override_cellsize=False,
                polarisation_frame=PolarisationFrame("stokesIQUV"),
            )
            pb = create_pb(model, telescope=telescope, use_local=False)
            if apply_pb:
                pbcomp = apply_beam_to_skycomponent(component, pb)
                log.debug(
                    "After application of primary beam {}".format(str(pbcomp.flux))
                )
            else:
                pbcomp = copy_skycomponent(component)
            bvis = dft_skycomponent_visibility(bvis, pbcomp)
            iquv_image = idft_visibility_skycomponent(bvis, component)[0]
            log.debug(
                "IQUV to {0} to IQUV image = {1}".format(vpol.type, iquv_image[0].flux)
            )

            if apply_pb:
                inverse_comp = apply_beam_to_skycomponent(pbcomp, pb, inverse=True)
            else:
                inverse_comp = copy_skycomponent(pbcomp)

            assert numpy.array_equal(component.flux, inverse_comp.flux)

    def test_apply_voltage_pattern_dft(self):
        self.createVis()
        nfailures = 0
        telescope = "MID_FEKO_B2"
        for flux in (
            numpy.array([[100.0, 0.0, 0.0, 0.0]]),
            numpy.array([[100.0, 100.0, 0.0, 0.0]]),
            numpy.array([[100.0, 0.0, 100.0, 0.0]]),
            numpy.array([[100.0, 0.0, 0.0, 100.0]]),
            numpy.array([[100.0, 1.0, -10.0, +60.0]]),
        ):
            vpol = PolarisationFrame("linear")
            try:
                bvis = create_blockvisibility(
                    self.config,
                    self.times,
                    self.frequency,
                    channel_bandwidth=self.channel_bandwidth,
                    phasecentre=self.phasecentre,
                    weight=1.0,
                    polarisation_frame=vpol,
                )

                component_centre = SkyCoord(
                    ra=+16.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
                )
                component = create_skycomponent(
                    direction=component_centre,
                    flux=flux,
                    frequency=self.frequency,
                    polarisation_frame=PolarisationFrame("stokesIQUV"),
                )
                model = create_image_from_visibility(
                    bvis,
                    cellsize=self.cellsize,
                    npixel=self.npixel,
                    override_cellsize=False,
                    polarisation_frame=PolarisationFrame("stokesIQUV"),
                )
                vpbeam = create_vp(model, telescope=telescope, use_local=False)
                vpbeam_wcs = vpbeam.image_acc.wcs
                vpbeam_wcs.wcs.ctype[0] = "RA---SIN"
                vpbeam_wcs.wcs.ctype[1] = "DEC--SIN"
                vpbeam_wcs.wcs.crval[0] = model.image_acc.wcs.wcs.crval[0]
                vpbeam_wcs.wcs.crval[1] = model.image_acc.wcs.wcs.crval[1]
                vpbeam = create_image_from_array(
                    vpbeam["pixels"],
                    wcs=vpbeam_wcs,
                    polarisation_frame=vpbeam.image_acc.polarisation_frame,
                )
                assert component.polarisation_frame == PolarisationFrame("stokesIQUV")
                vpcomp = apply_voltage_pattern_to_skycomponent(component, vpbeam)
                assert vpcomp.polarisation_frame == vpbeam.image_acc.polarisation_frame
                bvis = dft_skycomponent_visibility(bvis, vpcomp)
                vpcomp = idft_visibility_skycomponent(bvis, vpcomp)[0][0]
                assert (
                    vpcomp.polarisation_frame
                    == bvis.blockvisibility_acc.polarisation_frame
                )
                inv_vpcomp = apply_voltage_pattern_to_skycomponent(
                    vpcomp, vpbeam, inverse=True
                )
                assert vpcomp.polarisation_frame == PolarisationFrame("stokesIQUV")
                # print("After application of primary beam {}".format(str(vpcomp.flux)))
                # print("After correction of primary beam {}".format(str(inv_vpcomp.flux)))
                # print("{0} {1} succeeded".format(vpol, str(flux)))
                assert_array_almost_equal(flux, numpy.real(inv_vpcomp.flux), 9)
            except AssertionError as e:
                print(e)
                print("{0} {1} failed".format(vpol, str(flux)))
                nfailures += 1
        assert nfailures == 0, "{} tests failed".format(nfailures)


if __name__ == "__main__":
    unittest.main()
