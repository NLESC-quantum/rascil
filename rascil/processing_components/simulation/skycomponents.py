"""
Functions to deal with skycomponents in simulations.

"""

__all__ = ["addnoise_skycomponent"]

import logging
import collections
from typing import Union, List
import numpy
from numpy.random import default_rng

import astropy.units as u
from astropy.coordinates import SkyCoord
from rascil.data_models import Skycomponent
from rascil.data_models.polarisation import PolarisationFrame

from rascil.processing_components.skycomponent.operations import create_skycomponent

log = logging.getLogger("rascil-logger")


def addnoise_skycomponent(
    sc: Union[Skycomponent, List[Skycomponent]],
    noise=1e-3,
    mode="flux_central",
    seed=None,
) -> Union[Skycomponent, List[Skycomponent]]:
    """Add noise to Skycomponent

    :param sc: Skycomponent or list of skycomponents
    :param noise: Standard deviation of the distribution
    :param mode: Add noise to direction, flux_central, flux_all
    :param seed: Seed to generate noise
    :return: Skycomponent(s) with noise added
    """

    if seed is None:
        rng = default_rng(1805550721)
    else:
        rng = default_rng(seed)

    single = not isinstance(sc, collections.abc.Iterable) or isinstance(sc, str)

    if single:
        sc = [sc]

    log.debug("addnoise_skycomponent: Processing %d components" % (len(sc)))

    ras = [comp.direction.ra.radian for comp in sc]
    decs = [comp.direction.dec.radian for comp in sc]
    fluxes = numpy.array([comp.flux for comp in sc])

    comps = []

    if mode == "direction":

        ras += rng.normal(0.0, noise, len(ras))
        decs += rng.normal(0.0, noise, len(decs))

        new_directions = SkyCoord(ras * u.rad, decs * u.rad, frame="icrs")

        for i, direction in enumerate(new_directions):
            comps.append(
                Skycomponent(
                    direction=direction,
                    frequency=sc[i].frequency,
                    name="",
                    flux=sc[i].flux,
                    shape="Point",
                    polarisation_frame=sc[i].polarisation_frame,
                    params={},
                )
            )

    elif mode == "flux_central":

        # Only add to the central flux or single flux if single-frequency sc

        for i, flux in enumerate(fluxes):
            nchan = sc[i].frequency.shape[0]
            centre = nchan // 2
            flux[centre, 0] += rng.normal(0.0, noise, 1)
            comps.append(
                Skycomponent(
                    direction=sc[i].direction,
                    frequency=sc[i].frequency,
                    name="",
                    flux=flux,
                    shape="Point",
                    polarisation_frame=sc[i].polarisation_frame,
                    params={},
                )
            )

    elif mode == "flux_all":

        # add to all fluxes
        fluxes += rng.normal(0.0, noise, fluxes.shape)

        for i, flux in enumerate(fluxes):
            comps.append(
                Skycomponent(
                    direction=sc[i].direction,
                    frequency=sc[i].frequency,
                    name="",
                    flux=flux,
                    shape="Point",
                    polarisation_frame=sc[i].polarisation_frame,
                    params={},
                )
            )

    else:

        log.debug("Wrong mode")
        comps = sc

    return comps
