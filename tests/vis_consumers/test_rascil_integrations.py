import asyncio
import os
import time
import tempfile
import unittest

import pytest
from cbf_sdp import packetiser
from pytest_bdd.scenario import scenarios
from pytest_bdd.steps import given, then, when
from realtime.receive.core import ms_asserter, sched_tm, utils
from realtime.receive.core.config import create_config_parser

from realtime.receive.modules import receivers
try:
    from rascil.vis_consumer import msconsumer
except ImportError:
    raise ImportError(
        "RASCIL consumer not found"
        )


scenarios("./YAN-982.feature")

INPUT_FILE = "tests/vis_consumer/data/AA05LOW.ms"
SCHED_FILE = "tests/vis_consumer/data/sb-test.json"
OUTPUT_FILE = tempfile.mktemp(suffix=".ms", prefix="output_")

NUM_STREAMS = 96
CHAN_PER_STREAM = 144


@pytest.fixture(name="loop")
def get_loop():
    return asyncio.new_event_loop()


@given("An example input file of the correct dimension")
def test_file():
    if os.path.isdir(INPUT_FILE):
        return
    else:
        raise FileExistsError


@given("A scheduling block is available")
def test_file():
    if os.path.isfile(SCHED_FILE):
        return
    else:
        raise FileExistsError

@given(
    "A receiver can be configured with a RASCIL consumer",
    target_fixture="mswriter",
)
def get_receiver(loop):
    tm = sched_tm.SchedTM(SCHED_FILE)
    config = create_config_parser()
    config["reception"] = {
        "method": "spead2_receivers",
        "receiver_port_start": 42001,
        "consumer": "rascil.vis_consumer.msconsumer.consumer",
        "schedblock": SCHED_FILE,
        "outputfilename": OUTPUT_FILE,
        "ring_heaps": 128,
    }
    config["transmission"] = {
        "method": "spead2_transmitters",
        "target_host": "127.0.0.1",
        "target_port_start": str(42001),
        "channels_per_stream": str(CHAN_PER_STREAM),
    }
    config["reader"] = {"num_repeats": str(10), "num_timestamps": str(2)}

    return receivers.create(config, tm, loop)


@when("the data is sent at a rate commensurate with AA0.5")
def send_data(mswriter, loop):

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

    sending = packetiser.packetise(config, INPUT_FILE)

    time.sleep(5)

    # Go, go, go!
    async def run():
        coros = [sending, mswriter.run()]
        done, waiting = await asyncio.wait(coros, timeout=30)
        assert len(done) == len(coros)
        assert not waiting

    loop.run_until_complete(run())


@then("The same data is received and written")
def compare_measurement_sets():
    asserter = type(
        "asserter", (ms_asserter.MSAsserter, unittest.TestCase), {}
    )()
    asserter.assert_ms_data_equal(INPUT_FILE, OUTPUT_FILE)


@then("It is received without loss")
def received_ok(mswriter):
    assert mswriter.num_incomplete == 0, f"Failed to send data without loss"
