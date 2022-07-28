""" Unit tests for visibility operations


"""
import logging
import sys
import unittest

import xarray

from rascil.data_models import rascil_data_path
from rascil.processing_components.visibility.base import create_blockvisibility_from_ms

log = logging.getLogger("rascil-logger")

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))


class TestConcat(unittest.TestCase):
    def setUp(self):

        try:
            from casacore.tables import table  # pylint: disable=import-error

            self.casacore_available = True
        #            except ModuleNotFoundError:
        except:
            self.casacore_available = False

    def test_groupby(self):
        ms_list = ["vis/3C277.1C.16channels.ms", "vis/ASKAP_example.ms", "vis/xcasa.ms"]

        for ms in ms_list:
            for dim in ["frequency", "time", "polarisation"]:
                vis = create_blockvisibility_from_ms(rascil_data_path(ms))[0]

                # Don't squeeze out the unit dimensions because we will want
                # them for the concat
                # need to call .to_native_dataset() because else xarray.concat fails
                chan_vis = [
                    v[1].to_native_dataset() for v in vis.groupby(dim, squeeze=False)
                ]

                # Now concatenate
                newvis = xarray.concat(chan_vis, dim=dim, data_vars="minimal")

                # The xes may have been reordered so we align the objects. In testing,
                # this only affected the polarisation axis for the xcasa.ms
                newvis = xarray.align(vis, newvis)[1]

                assert newvis.equals(vis), "{}: {} Original {}\n\nRecovered {}".format(
                    ms, dim, vis, newvis
                )

    def test_groupby_bins(self):
        ms_list = ["vis/3C277.1C.16channels.ms", "vis/ASKAP_example.ms", "vis/xcasa.ms"]

        for ms in ms_list:
            for dim in ["frequency", "time"]:
                vis = create_blockvisibility_from_ms(rascil_data_path(ms))[0]

                # Don't squeeze out the unit dimensions because we will want
                # them for the concat
                # need to call .to_native_dataset() because else xarray.concat fails
                chan_vis = [
                    v[1].to_native_dataset() for v in vis.groupby_bins(dim, bins=2)
                ]

                # Now concatenate
                newvis = xarray.concat(chan_vis, dim=dim, data_vars="minimal")

                # The xes may have been reordered so we align the objects. In testing,
                # this only affected the polarisation axis for the xcasa.ms
                newvis = xarray.align(vis, newvis)[1]

                assert newvis.equals(vis), "{}: {} Original {}\n\nRecovered {}".format(
                    ms, dim, vis, newvis
                )


if __name__ == "__main__":
    unittest.main()
