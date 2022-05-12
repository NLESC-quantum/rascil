# -*- coding: utf-8 -*-


""" Takes a SPEAD2 HEAP and writes it to a MEASUREMENT SET. This is pretty much the
        same functionality as presented in the  OSKAR python binding example available at:
        https://github.com/OxfordSKA/OSKAR/blob/master/python/examples/spead/receiver/spead_recv.py
"""
import numpy
import numpy as np
from astropy.coordinates import SkyCoord, Angle
from astropy import units
import concurrent.futures
import logging
from configparser import ConfigParser
from datetime import datetime
from pathlib import Path
from rascil.data_models.memory_data_models import BlockVisibility, Configuration
from rascil.data_models.polarisation import PolarisationFrame
from typing import List, Optional

try:
    from overrides import overrides
    from realtime.receive.core import msutils
    from realtime.receive.core.icd import Payload, icd_to_ms
    from realtime.receive.core.sched_tm import SchedTM
    from realtime.receive.core.base_tm import BaseTM

    from realtime.receive.modules.consumers.iconsumer import IConsumer
    from realtime.receive.modules.utils.command_executor import CommandExecutor
except ImportError:
    raise ImportError(
        "SKA SDP Realtime Receive Modules is not installed; cannot"
        "operate visibility receive consumer. To install, run:\n"
        "pip install --extra-index-url=https://artefact.skao.int/repository/pypi-all/simple "
        "ska-sdp-realtime-receive-modules"
    )

logger = logging.getLogger(__name__)

def rcal_pipeline_start(Block: BlockVisibility):
    """
    THis is the rcal method that is called with a full BlockVisibility
    """

class VisibilityBucket(object):
    """
    Class to hold the visibilities for a single timestep until the buffers are full. This is needed as in the general case
    there are multiple input streams - each containing frequency slices (and perhaps even baseline slices)
    So we need to buffer them up.
    TODO: WHat to do if the streams arrive aout of order .... say the next time stamp arrives before the next is full

    """
    def __init__(self, model: SchedTM):
        """
        The full block of visibilities
        """
        self._model = model
        shape = (model.num_baselines,model.num_channels,model.num_pols)
        self._nant = len(self._model.get_antennas())
        uvw_matrix_shape = (self._nant,self._nant, 3)
        self._full_count = model.num_baselines * model.num_channels * model.num_pols
        self._visibilities = numpy.zeros(shape=shape, dtype=complex)
        self._uvw = numpy.zeros(shape=uvw_matrix_shape, dtype=float)
        self._flag = numpy.zeros(shape=shape, dtype=int)
        self._weight = numpy.ones(shape=shape,dtype=float)
        self._gauge = numpy.zeros_like(self._flag)
        self._is_full = False

    def set_time(self,time):
        """
        :param time: MJD seconds for this block
        """
        self._time = time

    def add_uvw(self, uvw):
        """
        Add the uvw for this buffer - needs to only be done once as all the frequency
        channels have the same UVW
        """
        """
        need to reshape as the uvw supplied have no redundancies and this seems to need a square
        """
        uvw_index = 0

        for ant1 in range(0,self._nant):
            for ant2 in range(ant1,self._nant):
                self._uvw[ant1,ant2] = uvw[uvw_index]
                self._uvw[ant2,ant1] = uvw[uvw_index]
                uvw_index = uvw_index + 1



    def add_visibilities(self, vis_slice, start_chan, num_chan):
        """
        Adds a slice to the bucket
        visibilities come from the payload - but they are in the wrong order
        THe ICD stacks then channels in freq x baseline x pol order - but the measurement
        sets want baseline x freq x pol - so we need to moveaxis

        """

        vis_to_add = icd_to_ms(vis_slice)
        stop_chan = start_chan+num_chan
        gauge_shape = (self._model.num_baselines, num_chan, self._model.num_pols)
        if numpy.shape(self._visibilities[:,start_chan:stop_chan]) != numpy.shape(vis_to_add):
            raise RuntimeError(
                f"Shape missmatch between slices input:{numpy.shape(vis_to_add)} "
                f"output:{numpy.shape(self._visibilities[:,start_chan:stop_chan])}"
            )

        if self._gauge[:,start_chan:stop_chan].sum() != 0:
            raise RuntimeError(
                "Trying to add a slice that is already present"
            )

        self._visibilities[:, start_chan:stop_chan] = vis_to_add
        self._gauge[:, start_chan:stop_chan] = numpy.ones(gauge_shape, dtype=int)
        self._check()

    def empty(self):
        """
        Empty bucket
        """
        self._gauge.fill(0.0)
        self._visibilities.fill(complex(0.0))
        self._flag.fill(0)
        self._weight.fill(0.0)


    def is_full(self) -> bool:
        """
        Is the Bucket full
        """
        self._check()
        return self._is_full

    def _check(self):
        """
        Need to check whether all the vis have been written for this Block
        TODO: How do we do this>>>
        """
        """
        if we have ascetained the bucket is full then
        """
        if self._gauge.sum() == self._full_count:
            self._is_full = True

    def get_visibilities(self):
        """
        return the VISIBILITY array
        """
        return self._visibilities

    def get_uvw(self):
        """
        return the UVW array
        """
        return self._uvw

    def get_time(self):
        """
        MJD time in seconds
        """
        return self._time

    def get_weight(self):
        """
        Return weight - should probably be correlation fraction?
        """
        return self._weight

class consumer(IConsumer):
    """
    A heap consumer that writes incoming data into an MS.

    Because data consumption happens inside the event loop we need to defer the
    data writing to a different thread. We do this by creating a single-threaded
    executor that we then use to schedule the I/O-heavy MS writing tasks onto.
    """

    @overrides
    def __init__(self, config: ConfigParser, tm: SchedTM):
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
        self._input_buffer = [VisibilityBucket(self.tm), VisibilityBucket(self.tm)]
        self._current_buffer = 0

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
        await self._buffer_payload(payload)
        self.received_payloads += 1

        # Write output ms if max payloads reached
        if (
            self.max_payloads is not None
            and self.received_payloads >= self.max_payloads
        ):
            logger.info("Max payloads received")
            self._finish_writing()
            self.received_payloads = 0

        if self._input_buffer[self._current_buffer].is_full():

            full_block_vis = self._fill_Block_Visibility(self.tm, self._input_buffer[self._current_buffer] )
            self._current_buffer = self._current_buffer+1
            self._current_buffer = self._current_buffer%2
            self._input_buffer[self._current_buffer].empty()

            """
            And go!!!- there is another buffer to fill - if this could be done asynchronously then
            you would get that extra time ....
            """
            rcal_pipeline_start(full_block_vis)


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

    def _init_Configuration(self, model: SchedTM) -> Configuration:

        """ Simple method to initialise the Configuration object describing data for processing

                name: Name of configuration e.g. 'LOWR3'
                location: Location of array as an astropy EarthLocation
                names: Names of the dishes/stations
                xyz: Geocentric coordinates of dishes/stations
                mount: Mount types of dishes/stations 'altaz' | 'xy' | 'equatorial'
                frame: Reference frame of locations
                receptor_frame: Receptor frame
                diameter: Diameters of dishes/stations (m)
                offset: Axis offset (m)
                stations: Identifiers of the dishes/stations
                vp_type: Type of voltage pattern (string)
        """
        antennas = model.get_antennas()
        """
        TODO: need to get the name from som
        """
        name = "unknown"
        """
        TODO: location not currently in the layout files
        """
        location = "unknown"
        """
        TODO: Check this is the kind of format you need
        TODO: offset not currently in layout
        """
        xyz = []
        diameter = []
        names = []
        stations = []
        offset = []
        for antenna in antennas:
            x_location = antennas[antenna]["x"]
            y_location = antennas[antenna]["y"]
            z_location = antennas[antenna]["z"]
            diameter.append(antennas[antenna]["dish_diameter"])
            stations.append(antennas[antenna]["name"])
            names.append(antennas[antenna]["name"])
            offset.append([0.0,0.0,0.0])
            xyz.append([x_location,y_location,z_location])

        """
        TODO: Mount not currently in the layout
        """
        mount = "unknown"
        """
        TODO: frame is often listed as ITRF - but in many cases is actually in WGS84
        """
        frame = "wgs84"
        """
        TODO: Receptor frame not listed in the layout
        """
        receptor_frame = "unknown"
        """
        TODO: vp_type is not known
        """
        vp_type = "unknown"


        return Configuration(
            name=name,
            location=location,
            names=names,
            xyz=xyz,
            mount=mount,
            frame=frame,
            receptor_frame=receptor_frame,
            diameter=diameter,
            offset=offset,
            stations=stations,
            vp_type=vp_type
        )

    def _fill_Block_Visibility(self, model: SchedTM,
                               buffer: VisibilityBucket) -> BlockVisibility:
        """
        Simple method to initialise contents of the BlockVisibility Object
        """

        frequency = []
        channel_bandwidth = []
        for chan in range(0, model.num_channels):
            frequency.append(model.freq_start_hz + chan*model.freq_inc_hz)
            channel_bandwidth.append(model.freq_inc_hz)


        """ 
        In the model we are holding the phase centre as ra-dec in rad.
        TODO: Make sure there is some security around the frame:
        
        """
        ra_rad, dec_rad = model.phase_centre_radec_rad
        target_ra = Angle(ra_rad*units.rad)
        target_dec = Angle(dec_rad*units.rad)
        phasecentre=SkyCoord(ra=target_ra, dec=target_dec)
        configuration=self._init_Configuration(model)

        """
        
        """

        vis=buffer.get_visibilities()
        uvw=buffer.get_uvw()
        time=buffer.get_time()
        weight=buffer.get_weight()
        integration_time=None
        flags=None
        baselines=None

        """
        TODO: Frame information not held
        """
        polarisation_frame=PolarisationFrame("linear")
        imaging_weight=None
        source="anonymous"
        meta=None
        low_precision="float64"

        return BlockVisibility(
            frequency=frequency,
            channel_bandwidth=channel_bandwidth,
            phasecentre=phasecentre,
            configuration=configuration,
            uvw=uvw,
            time=time,
            vis=vis,
            weight=weight,
            integration_time=integration_time,
            flags=flags,
            baselines=baselines,
            polarisation_frame=polarisation_frame,
            imaging_weight=imaging_weight,
            source=source,
            meta=meta,
            low_precision=low_precision
        )


    async def _buffer_payload(self, payload: Payload):

        """Writes a payload into a memory using the TM as the source of metadata"""
        # Find out the time index and TM data for this timestamp

        payload_time = payload.mjd_time

        """
               The UVW vectors for the vis are calculated based upon the time
        """

        time = payload.mjd_time
        uvw = []

        try:
            time_idx, data = self.tm.get_nearest_data(time)
            for index, u in enumerate(data.uu):
                v = data.vv[index]
                w = data.ww[index]
                uvw_vec = [u, v, w]
                uvw.append(uvw_vec)

        except ValueError:
            raise ValueError(f"No model time to match {payload_time}")

        assert len(data.uu) == self.tm.num_baselines, (
            f"Miss-match between baselines in heap {self.tm.num_baselines} "
            f"and uvw vector {len(data.uu)}"
        )

        vis = payload.visibilities
        first_chan = payload.channel_id
        chan_count = payload.channel_count

        buff = self._input_buffer[self._current_buffer]
        buff.add_visibilities(vis_slice=vis, start_chan=first_chan, num_chan=chan_count)
        buff.add_uvw(uvw=uvw)
        buff.set_time(time)




