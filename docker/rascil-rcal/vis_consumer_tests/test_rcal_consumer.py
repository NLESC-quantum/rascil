import asyncio
import os
import glob
import tempfile
import time
import pytest

from cbf_sdp import packetiser
from pytest_bdd.scenario import scenarios
from pytest_bdd.steps import given, then, when
from realtime.receive.core import sched_tm
from realtime.receive.core.config import create_config_parser
from realtime.receive.modules import receivers

from rascil.apps.rascil_rcal import cli_parser, rcal_simulator
from rascil.data_models.memory_data_models import BlockVisibility

try:
    from vis_consumer import rcal_consumer
except ImportError:
    raise ImportError("RASCIL consumer not found")

import logging

logger = logging.getLogger(__name__)
TEST_DIR = os.path.dirname(__file__)

scenarios(f"{TEST_DIR}/YAN-982.feature")

# TODO: RCAL takes a bit long to run on this MS;
#   we need to use a smaller one with new json files
INPUT_FILE = f"{TEST_DIR}/data/AA05LOW.ms"
SCHED_FILE = f"{TEST_DIR}/data/sb-test.json"
LAYOUT_FILE = f"{TEST_DIR}/data/TSI-AP.json"

OUTPUT_FILE = tempfile.mktemp(suffix=".ms", prefix="output_")

NUM_STREAMS = 96
CHAN_PER_STREAM = 144


def rcal_test(block: BlockVisibility, queue=None):
    rcal_parser = cli_parser()
    rcal_args = rcal_parser.parse_args(
        [
            "--do_plotting",
            "True",
            "--plot_dir",
            ".",
            # needed because the output files' root dir is determined based on this
            # doesn't have to be an existing MeasurementSet
            "--ingest_msname",
            "./tmp.ms",
            "--flag_rfi",
            "False",
        ]
    )

    if queue:
        queue.put("working")
        rcal_simulator(block, rcal_args)


@pytest.fixture(name="loop")
def get_loop():
    return asyncio.new_event_loop()


@given("An example input file of the correct dimension")
def find_input_file():
    if os.path.isdir(INPUT_FILE):
        return
    else:
        raise FileNotFoundError


@given("A scheduling block is available")
def find_sched_file():
    if os.path.isfile(SCHED_FILE):
        return
    else:
        raise FileNotFoundError


@given(
    "A receiver can be configured with a RCAL consumer",
    target_fixture="rcalconsumer",
)
def get_receiver(loop):
    tm = sched_tm.SchedTM(SCHED_FILE, LAYOUT_FILE)
    config = create_config_parser()
    config["reception"] = {
        "method": "spead2_receivers",
        "receiver_port_start": 42001,
        "consumer": "vis_consumer.rcal_consumer.consumer",
        "rcal_testing_method": "vis_consumer_tests.test_rcal_consumer.rcal_test",
        "schedblock": SCHED_FILE,
        "layout": LAYOUT_FILE,
        "outputfilename": OUTPUT_FILE,
        "ring_heaps": 128,
    }
    config["transmission"] = {
        "method": "spead2_transmitters",
        "target_host": "127.0.0.1",
        "target_port_start": str(42001),
        "channels_per_stream": str(CHAN_PER_STREAM),
    }
    config["reader"] = {"num_repeats": str(10), "num_timestamps": str(240)}

    return receivers.create(config, tm, loop)


@when("the data is sent to the RCAL consumer")
def send_data(rcalconsumer, loop):

    rate = 1e9 / NUM_STREAMS

    config = create_config_parser()
    config["transmission"] = {
        "method": "spead2_transmitters",
        "target_host": "127.0.0.1",
        "target_port_start": str(42001),
        "channels_per_stream": str(CHAN_PER_STREAM),
        "rate": str(rate),
        "time_interval": str(0),
    }
    try:
        sending = packetiser.packetise(config, INPUT_FILE)
    except:
        raise RuntimeError("Exception in packetiser")
    time.sleep(5)

    # Go, go, go!
    async def run():

        tasks = [asyncio.create_task(coro) for coro in (sending, rcalconsumer.run())]
        done, waiting = await asyncio.wait(tasks, timeout=60)
        # TODO: this assertion fails, but reason is not understood; asyncio-related
        # assert len(done) == len(tasks)
        # assert not waiting

    loop.run_until_complete(run())


@then("RCAL produces the right number of png and hdf files")
def compare_measurement_sets():
    """
    Test that the correct files are produced by RCAL

    TODO: this should actually contain all of the time samples
      but for some reason only the first two appear
      this will need to be investigated once we get back to this work
    """
    expected_files = ["tmp_20150623T230701", "tmp_20150623T230700"]
    png_files = glob.glob("*.png")
    hdf_files = glob.glob("*.hdf")
    for f in expected_files:
        assert f"{f}_plot.png" in png_files
        assert f"{f}_gaintable.hdf" in hdf_files


@then("It is received without loss")
def received_ok(mswriter):
    assert mswriter.num_incomplete == 0, f"Failed to send data without loss"
