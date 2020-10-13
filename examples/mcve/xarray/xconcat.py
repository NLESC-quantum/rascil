""" Test to illustrate delayed/concat clahese under xarray


Tim Cornwell 13 October 2020
realtimcornwell@gmail.com
"""
import numpy
from dask import delayed
from distributed import Client

import xarray

if __name__ == '__main__':

    client = Client(n_workers=4, threads_per_worker=4)
    print(client)
    import dask
    dask.config.set(scheduler="distributed")

    xar = xarray.DataArray(numpy.ones([16, 128, 128]), dims=["z", "y", "x"])

    def make_break_fix(x, i):
        lxar = [ar[1] for ar in x.groupby("z")]
        rec_x = xarray.concat(lxar, "z")
        assert rec_x.equals(x), i
        return rec_x
    
    graph = make_break_fix(xar, 0)
    graph = graph.chunk(chunks={"x": 64, "y":64})
    for iteration in range(16384):
        graph = make_break_fix(graph, iteration)
    
    exit(0)

    graph = delayed(make_break_fix(xar, 0))
    for iteration in range(16384):
        graph = delayed(make_break_fix, nout=1)(graph, iteration)
    new_xar = client.compute(graph, sync=True)
    
    exit()
