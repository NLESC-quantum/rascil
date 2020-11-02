""" Test to illustrate delayed/concat clahese under xarray


Tim Cornwell 13 October 2020
realtimcornwell@gmail.com
"""
import numpy
from dask import delayed
from distributed import Client

import xarray

if __name__ == '__main__':

    # client = Client(n_workers=4, threads_per_worker=4)
    # print(client)
    # import dask
    # dask.config.set(scheduler="distributed")

    original_xar = xarray.DataArray(numpy.ones([16, 1024, 1024]), dims=["z", "y", "x"]).chunk((4, 256, 256))
    original_xar2 = xarray.DataArray(numpy.ones([16, 512, 512]), dims=["z", "y", "x"])
    print(original_xar)
    print(original_xar.data)

    def scatter_z(x):
        for ar in x.groupby("z"):
            selection = {"x": slice(256,768), "y": slice(256,768)}
            yield ar[1].isel(selection)
            
    def gather_z(lx):
        return xarray.concat(lx, "z")
    
    def check(xar, rec_x):
        print(xar, rec_x)
        return rec_x.equals(xar)
        
    future_xar = original_xar.persist()
    list_x = delayed(scatter_z)(future_xar)
    rec_x = delayed(gather_z)(list_x)
    
    one_pass_graph = delayed(check)(original_xar2, rec_x)
    one_pass_result = one_pass_graph.compute()
    assert one_pass_result
    exit()
