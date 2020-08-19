
import casacore.tables as tables

t = tables.table("ms_write.ms")

import pprint
pp = pprint.PrettyPrinter()
pp.pprint(t.getdesc())
