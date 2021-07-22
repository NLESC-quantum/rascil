""" Unit tests for image Taylor terms

"""
import logging
import os
import unittest

import numpy


from astropy.coordinates import SkyCoord
import astropy.units as u

from rascil.processing_components.skycomponent.taylor_terms import (
    calculate_frequency_taylor_terms_from_skycomponents,
    find_skycomponents_frequency_taylor_terms,
)

from rascil.processing_components import (
    create_low_test_skycomponents_from_gleam,
    create_low_test_image_from_gleam,
    copy_skycomponent,
    smooth_image,
)

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)


class TestSkycomponentTaylorTerm(unittest.TestCase):
    def setUp(self):

        from rascil.data_models.parameters import rascil_path

        self.dir = rascil_path("test_results")

        self.persist = os.getenv("RASCIL_PERSIST", False)

    def test_calculate_taylor_terms(self):
        phasecentre = SkyCoord(
            ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000"
        )
        frequency = numpy.linspace(0.9e8, 1.1e8, 9)
        sc = create_low_test_skycomponents_from_gleam(
            phasecentre=phasecentre, frequency=frequency, flux_limit=10.0
        )[0]
        sc_list = []
        for chan, frequency in enumerate(sc.frequency):
            newsc = copy_skycomponent(sc)
            newsc.frequency = numpy.array([frequency])
            newsc.flux = sc.flux[chan][numpy.newaxis, :]
            sc_list.append(newsc)

        taylor_term_list = calculate_frequency_taylor_terms_from_skycomponents(
            sc_list, nmoment=3
        )
        assert len(taylor_term_list) == 3

    def test_find_skycomponents_frequency_taylor_terms(self):
        phasecentre = SkyCoord(
            ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000"
        )
        frequency = numpy.linspace(0.9e8, 1.1e8, 9)
        im_list = [
            create_low_test_image_from_gleam(
                cellsize=0.001,
                npixel=512,
                phasecentre=phasecentre,
                frequency=[f],
                flux_limit=10.0,
            )
            for f in frequency
        ]
        im_list = [smooth_image(im, width=2.0) for im in im_list]

        for moment in [1, 2, 3]:
            sc_list = find_skycomponents_frequency_taylor_terms(
                im_list, nmoment=moment, flux_limit=20.0
            )
            print(moment, len(sc_list), len(sc_list[0]))


if __name__ == "__main__":
    unittest.main()
