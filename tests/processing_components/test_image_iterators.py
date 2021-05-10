"""Unit tests for image iteration


"""
import logging
import unittest

import numpy

from rascil.data_models import PolarisationFrame, rascil_path

from rascil.processing_components.image.iterators import (
    image_raster_iter,
    image_channel_iter,
    image_null_iter,
)

from rascil.processing_components.image.operations import (
    export_image_to_fits,
    pad_image,
)
from rascil.processing_components.simulation import create_test_image

log = logging.getLogger("rascil-logger")
log.setLevel(logging.WARNING)

log.setLevel(logging.WARNING)


class TestImageIterators(unittest.TestCase):
    def get_test_image(self, npixel=512):
        testim = create_test_image(polarisation_frame=PolarisationFrame("stokesI"))
        return pad_image(testim, [1, 1, npixel, npixel])

    def test_raster(self):
        """Test a raster iterator across an image. The test is to check that the
        value of the subimages is multiplied by two.

        """

        testdir = rascil_path("test_results")
        for npixel in [256, 512, 1024]:
            m31original = self.get_test_image(npixel=npixel)
            assert numpy.max(numpy.abs(m31original["pixels"].data)), "Original is empty"

            for nraster in [1, 4, 8, 16]:

                for overlap in [0, 2, 4, 8, 16]:
                    try:
                        m31model = self.get_test_image(npixel=npixel)
                        for patch in image_raster_iter(
                            m31model, facets=nraster, overlap=overlap
                        ):
                            assert patch["pixels"].data.shape[3] == (
                                m31model["pixels"].data.shape[3] // nraster
                            ), "Number of pixels in each patch: %d not as expected: %d" % (
                                patch["pixels"].data.shape[3],
                                (m31model["pixels"].data.shape[3] // nraster),
                            )
                            assert patch["pixels"].data.shape[2] == (
                                m31model["pixels"].data.shape[2] // nraster
                            ), "Number of pixels in each patch: %d not as expected: %d" % (
                                patch["pixels"].data.shape[2],
                                (m31model["pixels"].data.shape[2] // nraster),
                            )
                            patch["pixels"].data *= 2.0

                        if numpy.max(numpy.abs(m31model["pixels"].data)) == 0.0:
                            log.warning(
                                f"Raster is empty failed for {npixel}, {nraster}, {overlap}"
                            )
                        diff = m31model.copy(deep=True)
                        diff["pixels"].data -= 2.0 * m31original["pixels"].data
                        err = numpy.max(diff["pixels"].data)
                        if abs(err) > 0.0:
                            log.warning(
                                f"Raster set failed for {npixel}, {nraster}, {overlap}: error {err}"
                            )
                        export_image_to_fits(
                            m31model,
                            f"{testdir}/test_image_iterators_model_{npixel}_{nraster}_{overlap}.fits",
                        )
                        export_image_to_fits(
                            diff,
                            f"{testdir}/test_image_iterators_diff_{npixel}_{nraster}_{overlap}.fits",
                        )
                    except ValueError as err:
                        log.error(
                            f"Iterator failed for {npixel}, {nraster}, {overlap}, : {err}"
                        )

    def test_raster_exception(self):

        m31original = self.get_test_image()
        assert numpy.max(numpy.abs(m31original["pixels"].data)), "Original is empty"

        for nraster, overlap in [(-1, -1), (-1, 0), (1e6, 127)]:
            with self.assertRaises(AssertionError):
                m31model = create_test_image(
                    polarisation_frame=PolarisationFrame("stokesI")
                )
                for patch in image_raster_iter(
                    m31model, facets=nraster, overlap=overlap
                ):
                    patch["pixels"].data *= 2.0

        for nraster, overlap in [(2, 128)]:
            with self.assertRaises(ValueError):
                m31model = create_test_image(
                    polarisation_frame=PolarisationFrame("stokesI")
                )
                for patch in image_raster_iter(
                    m31model, facets=nraster, overlap=overlap
                ):
                    patch["pixels"].data *= 2.0

    def test_channelise(self):
        m31cube = create_test_image(
            frequency=numpy.linspace(1e8, 1.1e8, 128),
            polarisation_frame=PolarisationFrame("stokesI"),
        )

        for subimages in [128, 16, 8, 2, 1]:
            for slab in image_channel_iter(m31cube, subimages=subimages):
                assert slab["pixels"].data.shape[0] == 128 // subimages

    def test_null(self):
        m31cube = create_test_image(
            frequency=numpy.linspace(1e8, 1.1e8, 128),
            polarisation_frame=PolarisationFrame("stokesI"),
        )

        for i, im in enumerate(image_null_iter(m31cube)):
            assert i < 1, "Null iterator returns more than one value"


if __name__ == "__main__":
    unittest.main()
