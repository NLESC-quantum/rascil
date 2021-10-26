import sys

sys.path.append(".")
import unittest

# dir = '/Users/wangfeng/dev/rascil/data/vis/ASKAP_example.ms'

import logging
import numpy

from rascil.data_models.parameters import rascil_path, rascil_data_path

from rascil.processing_components import (
    create_image_from_visibility,
    predict_blockvisibility,
)

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))


run_ms_tests = False
try:
    import casacore
    from rascil.processing_components.visibility.base import (
        create_blockvisibility,
        create_blockvisibility_from_ms,
    )
    from rascil.processing_components.visibility.base import (
        export_blockvisibility_to_ms,
    )

    run_ms_tests = True
except ImportError:
    pass


@unittest.skipUnless(run_ms_tests, "requires the 'casacore' module")
class export_ms_RASCIL_test(unittest.TestCase):
    def setUp(self):
        """Turn off all numpy warnings and create the temporary file directory."""
        pass

    def test_copy_ms(self):
        if run_ms_tests == False:
            return

        msfile = rascil_data_path("vis/ASKAP_example.ms")
        msoutfile = rascil_path("test_results/test_export_ms_ASKAP_output.ms")

        v = create_blockvisibility_from_ms(msfile)
        export_blockvisibility_to_ms(
            msoutfile, v
        )  # vis_by_channel.append(integrate_visibility_by_channel(v[0]))

    def test_export_ms(self):
        if run_ms_tests == False:
            return

        msoutfile = rascil_path("test_results/test_export_ms_ASKAP_output.ms")

        from astropy.coordinates import SkyCoord
        from astropy import units as u

        from rascil.processing_components.simulation import create_named_configuration
        from rascil.processing_components.simulation import create_test_image
        from rascil.processing_components.imaging.base import (
            advise_wide_field,
        )

        from rascil.data_models.polarisation import PolarisationFrame

        lowr3 = create_named_configuration("LOWBD2", rmax=750.0)

        times = numpy.zeros([1])
        frequency = numpy.array([1e8])
        channelbandwidth = numpy.array([1e6])
        phasecentre = SkyCoord(
            ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000"
        )

        bvis = create_blockvisibility(
            lowr3,
            times,
            frequency,
            phasecentre=phasecentre,
            weight=1.0,
            polarisation_frame=PolarisationFrame("stokesI"),
            channel_bandwidth=channelbandwidth,
        )

        advice = advise_wide_field(
            bvis,
            guard_band_image=3.0,
            delA=0.1,
            facets=1,
            wprojection_planes=1,
            oversampling_synthesised_beam=4.0,
        )
        cellsize = advice["cellsize"]

        m31image = create_test_image(cellsize=cellsize, frequency=frequency)
        nchan, npol, ny, nx = m31image["pixels"].data.shape
        m31image = create_image_from_visibility(bvis, cellsize=cellsize, npixel=nx)
        bvis = predict_blockvisibility(bvis, m31image, context="2d")
        export_blockvisibility_to_ms(msoutfile, [bvis], source_name="M31")


class export_measurementset_test_suite(unittest.TestSuite):
    """A unittest.TestSuite class which tests exporting measurementset
    tests."""

    def __init__(self):
        unittest.TestSuite.__init__(self)

        loader = unittest.TestLoader()
        self.addTests(loader.loadTestsFromTestCase(export_ms_RASCIL_test))


if __name__ == "__main__":
    unittest.main()
