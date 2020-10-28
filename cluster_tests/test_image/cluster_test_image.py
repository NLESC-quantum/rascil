""" Cluster test


"""
import logging
import sys

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from distributed import Client

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.imaging.base import create_image_from_visibility
from rascil.processing_components.imaging.primary_beams import create_pb
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility.base import create_blockvisibility
from rascil.processing_components.visibility.coalesce import (
    convert_blockvisibility_to_visibility,
)
from rascil.workflows.rsexecute.image.image_rsexecute import (
    image_rsexecute_map_workflow,
)
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute

log = logging.getLogger()
log.setLevel(logging.DEBUG)

logging.getLogger("asyncio").setLevel(logging.WARNING)

if __name__ == "__main__":

    print("Starting cluster_test_image")
    # We pass in the scheduler from the invoking script
    if len(sys.argv) > 1:
        scheduler = sys.argv[1]
        client = Client(scheduler)
    else:
        client = Client()
    rsexecute.set_client(client=client)

    from rascil.data_models.parameters import rascil_path

    dir = rascil_path("test_results")

    frequency = numpy.linspace(1e8, 1.5e8, 3)
    channel_bandwidth = numpy.array([2.5e7, 2.5e7, 2.5e7])
    flux = numpy.array([[100.0], [100.0], [100.0]])
    config = create_named_configuration("LOWBD2-CORE")
    times = numpy.linspace(-300.0, 300.0, 3) * numpy.pi / 43200.0
    nants = config.xyz.shape[0]
    assert nants > 1
    assert len(config.names) == nants
    assert len(config.mount) == nants

    config = create_named_configuration("LOWBD2", rmax=1000.0)
    phasecentre = SkyCoord(
        ra=+15 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000"
    )

    bvis_graph = rsexecute.execute(create_blockvisibility)(
        config,
        times,
        frequency,
        channel_bandwidth=channel_bandwidth,
        phasecentre=phasecentre,
        weight=1.0,
        polarisation_frame=PolarisationFrame("stokesI"),
    )
    vis_graph = rsexecute.execute(convert_blockvisibility_to_visibility)(bvis_graph)

    model_graph = rsexecute.execute(create_image_from_visibility)(
        vis_graph, npixel=4096, cellsize=0.001, override_cellsize=False
    )
    beam = image_rsexecute_map_workflow(
        model_graph, create_pb, facets=16, pointingcentre=phasecentre, telescope="MID"
    )
    beam = rsexecute.compute(beam, sync=True)

    assert numpy.max(beam.data) > 0.0

    print("Successfully finished test_image")
    exit(0)
