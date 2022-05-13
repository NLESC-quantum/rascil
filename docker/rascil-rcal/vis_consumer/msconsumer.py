# -*- coding: utf-8 -*-


""" Takes a SPEAD2 HEAP and writes it to a MEASUREMENT SET. This is pretty much the
        same functionality as presented in the  OSKAR python binding example available at:
        https://github.com/OxfordSKA/OSKAR/blob/master/python/examples/spead/receiver/spead_recv.py
"""
import concurrent.futures
import logging
from configparser import ConfigParser
from datetime import datetime
from pathlib import Path
from typing import List, Optional

try:
    from overrides import overrides
    from realtime.receive.core import BaseTM, msutils
    from realtime.receive.modules.consumers.iconsumer import IConsumer
    from realtime.receive.modules.utils.command_executor import CommandExecutor
except ImportError:
    raise ImportError(
        "SKA SDP Realtime Receive Modules is not installed; cannot "
        "operate visibility receive consumer. To install, run:\n"
        "pip install --extra-index-url=https://artefact.skao.int/repository/pypi-all/simple "
        "ska-sdp-realtime-receive-modules"
    )

logger = logging.getLogger(__name__)


class consumer(IConsumer):
    """
    A heap consumer that writes incoming data into an MS.

    Because data consumption happens inside the event loop we need to defer the
    data writing to a different thread. We do this by creating a single-threaded
    executor that we then use to schedule the I/O-heavy MS writing tasks onto.
    """

    @overrides
    def __init__(self, config: ConfigParser, tm: BaseTM):
        self.outputfilename: str = config["reception"].get(
            "outputfilename", "recv-vis.ms"
        )
        self.max_payloads: Optional[int] = config["reception"].getoptint(
            "max_payloads", None
        )
        self._command_template: Optional[List[str]] = config["reception"].getoptlist(
            "command_template", None
        )
        self._timestamp_output: bool = config["reception"].getboolean(
            "timestamp_output", False
        )

        self.tm = tm
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.received_payloads = 0
        self._command_executor = None
        self.mswriter = None
        if self._command_template:
            self._command_executor = CommandExecutor(self._command_template)

    def _generate_output_path(self) -> str:
        if self._timestamp_output:
            # UTC Date Time Format
            p = Path(self.outputfilename)
            return f'{p.stem}.{str(datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))}{p.suffix}'
        else:
            return self.outputfilename

    @overrides
    async def consume(self, payload):
        """Entry point invoked by the receiver each time a heap arrives"""
        if self.mswriter is None:
            output_path = self._generate_output_path()
            logger.info(f"Writing to {output_path}")
            self.mswriter = msutils.MSWriter(output_path, self.tm)
        await self.mswriter.write_payload(payload, self.tm, executor=self.executor)
        self.received_payloads += 1

        # Write output ms if max payloads reached
        if (
            self.max_payloads is not None
            and self.received_payloads >= self.max_payloads
        ):
            logger.info("Max payloads received")
            self._finish_writing()
            self.received_payloads = 0

    def _finish_writing(self):
        if self.mswriter is None:
            raise Exception("mswriter doesn't exist")
        else:
            output_path = self.mswriter.ms.name
            self.mswriter.close()
            self.mswriter = None
            logger.info("Finished writing %s", output_path)
            if self._command_executor:
                self._command_executor.schedule(output_path)
                # if writing to the same output file, wait until the
                # command executor finishes before overwriting. Timestamp
                # output option does not perform overwriting.
                if not self._timestamp_output:
                    self._command_executor.stop()

    @overrides
    def stop(self):
        if self.mswriter is not None:
            self._finish_writing()
        if self._command_executor is not None:
            self._command_executor.stop()
