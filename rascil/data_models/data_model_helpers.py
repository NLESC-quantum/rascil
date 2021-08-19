"""Functions to help with persistence of data models

These do data conversion and persistence. Functions from processing_components are used.
"""

__all__ = [
    "convert_earthlocation_to_string",
    "convert_earthlocation_from_string",
    "convert_direction_to_string",
    "convert_direction_from_string",
    "convert_configuration_to_hdf",
    "convert_configuration_from_hdf",
    "convert_blockvisibility_to_hdf",
    "convert_hdf_to_blockvisibility",
    "convert_flagtable_to_hdf",
    "convert_hdf_to_flagtable",
    "export_blockvisibility_to_hdf5",
    "import_blockvisibility_from_hdf5",
    "convert_gaintable_to_hdf",
    "convert_hdf_to_gaintable",
    "export_gaintable_to_hdf5",
    "import_gaintable_from_hdf5",
    "convert_pointingtable_to_hdf",
    "convert_hdf_to_pointingtable",
    "export_pointingtable_to_hdf5",
    "import_pointingtable_from_hdf5",
    "convert_skycomponent_to_hdf",
    "convert_hdf_to_skycomponent",
    "export_skycomponent_to_hdf5",
    "import_skycomponent_from_hdf5",
    "convert_image_to_hdf",
    "convert_hdf_to_image",
    "export_image_to_hdf5",
    "import_image_from_hdf5",
    "export_skymodel_to_hdf5",
    "convert_skymodel_to_hdf",
    "import_skymodel_from_hdf5",
    "convert_hdf_to_skymodel",
    "convert_griddata_to_hdf",
    "convert_hdf_to_griddata",
    "export_griddata_to_hdf5",
    "import_griddata_from_hdf5",
    "convert_convolutionfunction_to_hdf",
    "convert_hdf_to_convolutionfunction",
    "export_convolutionfunction_to_hdf5",
    "import_convolutionfunction_from_hdf5",
    "memory_data_model_to_buffer",
    "buffer_data_model_to_memory",
    "convert_earthlocation_to_string",
    "convert_earthlocation_from_string",
    "convert_direction_to_string",
    "convert_direction_from_string",
    "convert_configuration_to_hdf",
    "convert_configuration_from_hdf",
    "convert_blockvisibility_to_hdf",
    "convert_hdf_to_blockvisibility",
    "convert_flagtable_to_hdf",
    "convert_hdf_to_flagtable",
    "export_blockvisibility_to_hdf5",
    "import_blockvisibility_from_hdf5",
    "export_flagtable_to_hdf5",
    "import_flagtable_from_hdf5",
    "convert_gaintable_to_hdf",
    "convert_hdf_to_gaintable",
    "export_gaintable_to_hdf5",
    "import_gaintable_from_hdf5",
    "convert_pointingtable_to_hdf",
    "convert_hdf_to_pointingtable",
    "export_pointingtable_to_hdf5",
    "import_pointingtable_from_hdf5",
    "convert_skycomponent_to_hdf",
    "convert_hdf_to_skycomponent",
    "export_skycomponent_to_hdf5",
    "import_skycomponent_from_hdf5",
    "convert_image_to_hdf",
    "convert_hdf_to_image",
    "export_image_to_hdf5",
    "import_image_from_hdf5",
    "export_skymodel_to_hdf5",
    "convert_skymodel_to_hdf",
    "import_skymodel_from_hdf5",
    "convert_hdf_to_skymodel",
    "convert_griddata_to_hdf",
    "convert_hdf_to_griddata",
    "export_griddata_to_hdf5",
    "import_griddata_from_hdf5",
    "convert_convolutionfunction_to_hdf",
    "convert_hdf_to_convolutionfunction",
    "export_convolutionfunction_to_hdf5",
    "import_convolutionfunction_from_hdf5",
    "memory_data_model_to_buffer",
    "buffer_data_model_to_memory",
    "data_model_equals",
]

import ast
import logging
import collections
from typing import Union

import astropy.units as u
import h5py
import numpy
import xarray
import pandas
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.units import Quantity
from astropy.wcs import WCS

from rascil.data_models.memory_data_models import (
    BlockVisibility,
    Configuration,
    GainTable,
    SkyModel,
    Skycomponent,
    Image,
    GridData,
    ConvolutionFunction,
    PointingTable,
    FlagTable,
)
from rascil.data_models.polarisation import PolarisationFrame, ReceptorFrame


log = logging.getLogger("rascil-logger")


def convert_earthlocation_to_string(el: EarthLocation):
    """Convert Earth Location to string

    :param el:
    :return:
    """
    return "%s, %s, %s" % (el.x, el.y, el.z)


def convert_earthlocation_from_string(s: str):
    """Convert Earth Location to string

    :param s: String

    :return:
    """
    x, y, z = s.split(",")
    el = EarthLocation(x=Quantity(x), y=Quantity(y), z=Quantity(z))
    return el


def convert_direction_to_string(d: SkyCoord):
    """Convert SkyCoord to string

    :param d: SkyCoord
    :return:
    """
    return "%s, %s, %s" % (d.ra.deg, d.dec.deg, "icrs")


def convert_direction_from_string(s: str):
    """Convert direction (SkyCoord) from string

    :param s: String

    :return:
    """
    ra, dec, frame = s.split(",")
    d = SkyCoord(ra, dec, unit="deg", frame=frame.strip())
    return d


def convert_configuration_to_hdf(config: Configuration, f):
    """Convert a Configuration to an HDF file

    :param config: Configuration
    :param f: hdf group
    :return: group with config added
    """
    if not isinstance(config, Configuration):
        raise ValueError(f"config is not a Configuration: {config}")
    if config.attrs["rascil_data_model"] != "Configuration":
        raise ValueError(f"config is not a Configuration: {config}")

    cf = f.create_group("configuration")
    cf.attrs["rascil_data_model"] = "Configuration"
    cf.attrs["name"] = config.name
    cf.attrs["location"] = convert_earthlocation_to_string(config.location)
    cf.attrs["frame"] = config.frame
    cf.attrs["receptor_frame"] = config.receptor_frame.type

    cf["configuration/xyz"] = config.xyz
    cf["configuration/diameter"] = config.diameter
    cf["configuration/names"] = [numpy.string_(name) for name in config.names]
    cf["configuration/mount"] = [numpy.string_(mount) for mount in config.mount]
    cf["configuration/offset"] = config.offset
    cf["configuration/stations"] = [
        numpy.string_(station) for station in config.stations
    ]
    cf["configuration/vp_type"] = [numpy.string_(vpt) for vpt in config.vp_type]
    return f


def convert_configuration_from_hdf(f):
    """Extract configuration from HDF

    :param f: hdf group
    :return: Configuration
    """
    cf = f["configuration"]

    assert cf.attrs["rascil_data_model"] == "Configuration", (
        "%s is a Configuration" % cf.attrs["rascil_data_model"]
    )

    name = cf.attrs["name"]
    location = convert_earthlocation_from_string(cf.attrs["location"])
    receptor_frame = ReceptorFrame(cf.attrs["receptor_frame"])
    frame = cf.attrs["frame"]

    xyz = cf["configuration/xyz"]
    diameter = cf["configuration/diameter"]
    names = [str(n) for n in cf["configuration/names"]]
    mount = [str(m) for m in cf["configuration/mount"]]
    stations = [str(p) for p in cf["configuration/stations"]]
    vp_type = [str(p) for p in cf["configuration/vp_type"]]
    offset = cf["configuration/offset"]
    return Configuration(
        name=name,
        location=location,
        receptor_frame=receptor_frame,
        xyz=xyz,
        frame=frame,
        diameter=diameter,
        names=names,
        mount=mount,
        offset=offset,
        stations=stations,
        vp_type=vp_type,
    )


def convert_blockvisibility_to_hdf(vis: BlockVisibility, f):
    """Convert a BlockVisibility to an HDF file

    :param vis: BlockVisibility
    :param f: hdf group
    :return: group with vis added
    """
    if not isinstance(vis, xarray.Dataset):
        raise ValueError("vis is not an xarray.Dataset")

    if vis.attrs["rascil_data_model"] != "BlockVisibility":
        raise ValueError(f"vis is not a BlockVisibility: {vis}")

    # We only need to keep the things we need to reconstruct the data_model
    f.attrs["rascil_data_model"] = "BlockVisibility"
    f.attrs["nants"] = numpy.max([b[1] for b in vis.baselines.data]) + 1
    f.attrs["nvis"] = vis.blockvisibility_acc.nvis
    f.attrs["npol"] = vis.blockvisibility_acc.npol
    f.attrs["phasecentre_coords"] = vis.phasecentre.to_string()
    f.attrs["phasecentre_frame"] = vis.phasecentre.frame.name
    f.attrs["polarisation_frame"] = vis.blockvisibility_acc.polarisation_frame.type
    f.attrs["source"] = vis.source
    f.attrs["meta"] = str(vis.meta)
    datavars = [
        "time",
        "frequency",
        "channel_bandwidth",
        "vis",
        "weight",
        "imaging_weight",
        "flags",
        "uvw",
        "integration_time",
    ]
    for var in datavars:
        f["data_{}".format(var)] = vis[var].data
    f = convert_configuration_to_hdf(vis.configuration, f)
    return f


def convert_hdf_to_blockvisibility(f):
    """Convert HDF root to blockvisibility

    :param f: hdf group
    :return: BlockVisibility
    """
    assert f.attrs["rascil_data_model"] == "BlockVisibility", "Not a BlockVisibility"
    s = f.attrs["phasecentre_coords"].split()
    ss = [float(s[0]), float(s[1])] * u.deg
    phasecentre = SkyCoord(ra=ss[0], dec=ss[1], frame=f.attrs["phasecentre_frame"])
    polarisation_frame = PolarisationFrame(f.attrs["polarisation_frame"])
    source = f.attrs["source"]
    nants = f.attrs["nants"]
    meta = ast.literal_eval(f.attrs["meta"])
    time = f["data_time"][()]
    frequency = f["data_frequency"][()]
    channel_bandwidth = f["data_channel_bandwidth"][()]
    uvw = f["data_uvw"][()]
    integration_time = f["data_integration_time"][()]
    vis = f["data_vis"][()]
    weight = f["data_weight"][()]
    imaging_weight = f["data_imaging_weight"][()]
    flags = f["data_flags"][()]

    from rascil.processing_components.visibility import generate_baselines

    baselines = pandas.MultiIndex.from_tuples(
        generate_baselines(nants), names=("antenna1", "antenna2")
    )

    vis = BlockVisibility(
        vis=vis,
        time=time,
        uvw=uvw,
        integration_time=integration_time,
        frequency=frequency,
        weight=weight,
        imaging_weight=imaging_weight,
        flags=flags,
        baselines=baselines,
        polarisation_frame=polarisation_frame,
        phasecentre=phasecentre,
        channel_bandwidth=channel_bandwidth,
        source=source,
        meta=meta,
        configuration=convert_configuration_from_hdf(f),
    )
    return vis


def convert_flagtable_to_hdf(ft: FlagTable, f):
    """Convert a FlagTable to an HDF file

    :param ft: FlagTable
    :param f: hdf group
    :return: group with ft added
    """
    if not isinstance(ft, xarray.Dataset):
        raise ValueError(f"ft is not an xarray.Dataset: {ft}")
    if ft.attrs["rascil_data_model"] != "FlagTable":
        raise ValueError(f"ft is not a FlagTable: {ft}")

    f.attrs["rascil_data_model"] = "FlagTable"
    f.attrs["nants"] = numpy.max([b[1] for b in ft.baselines.data]) + 1
    f.attrs["polarisation_frame"] = ft.flagtable_acc.polarisation_frame.type
    datavars = ["time", "frequency", "flags", "integration_time", "channel_bandwidth"]
    for var in datavars:
        f["data_{}".format(var)] = ft[var].data
    f = convert_configuration_to_hdf(ft.configuration, f)
    return f


def convert_hdf_to_flagtable(f):
    """Convert HDF root to flagtable

    :param f: hdf group
    :return: FlagTable
    """
    assert f.attrs["rascil_data_model"] == "FlagTable", "Not a FlagTable"
    nants = f.attrs["nants"]
    from rascil.processing_components import generate_baselines

    baselines = pandas.MultiIndex.from_tuples(
        generate_baselines(nants), names=("antenna1", "antenna2")
    )
    polarisation_frame = PolarisationFrame(f.attrs["polarisation_frame"])
    frequency = f["data_frequency"][()]
    channel_bandwidth = f["data_channel_bandwidth"][()]
    time = f["data_time"][()]
    flags = f["data_flags"][()]
    integration_time = f["data_integration_time"][()]
    ft = FlagTable(
        time=time,
        flags=flags,
        frequency=frequency,
        baselines=baselines,
        integration_time=integration_time,
        channel_bandwidth=channel_bandwidth,
        polarisation_frame=polarisation_frame,
        configuration=convert_configuration_from_hdf(f),
    )
    return ft


def export_blockvisibility_to_hdf5(vis, filename):
    """Export a BlockVisibility to HDF5 format

    :param vis:
    :param filename:
    :return:
    """

    if not isinstance(vis, collections.abc.Iterable):
        vis = [vis]
    with h5py.File(filename, "w") as f:
        if isinstance(vis, list):
            f.attrs["number_data_models"] = len(vis)
            for i, v in enumerate(vis):
                vf = f.create_group("BlockVisibility%d" % i)
                convert_blockvisibility_to_hdf(v, vf)
        else:
            f.attrs["number_data_models"] = 1
            vf = f.create_group("BlockVisibility%d" % 0)
            convert_blockvisibility_to_hdf(vis, vf)

        f.flush()


def import_blockvisibility_from_hdf5(filename):
    """Import BlockVisibility(s) from HDF5 format

    :param filename:
    :return: If only one then a BlockVisibility, otherwise a list of BlockVisibility's
    """

    with h5py.File(filename, "r") as f:
        nvislist = f.attrs["number_data_models"]
        vislist = [
            convert_hdf_to_blockvisibility(f["BlockVisibility%d" % i])
            for i in range(nvislist)
        ]
        if nvislist == 1:
            return vislist[0]
        else:
            return vislist


def export_flagtable_to_hdf5(ft, filename):
    """Export a FlagTable or list to HDF5 format

    :param ft:
    :param filename:
    """

    if not isinstance(ft, collections.abc.Iterable):
        ft = [ft]
    with h5py.File(filename, "w") as f:
        if isinstance(ft, list):
            f.attrs["number_data_models"] = len(ft)
            for i, v in enumerate(ft):
                vf = f.create_group("FlagTable%d" % i)
                convert_flagtable_to_hdf(v, vf)
        else:
            f.attrs["number_data_models"] = 1
            vf = f.create_group("FlagTable%d" % 0)
            convert_flagtable_to_hdf(ft, vf)
        f.flush()


def import_flagtable_from_hdf5(filename):
    """Import FlagTable(s) from HDF5 format

    :param filename:
    :return: If only one then a FlagTable, otherwise a list of FlagTable's
    """

    with h5py.File(filename, "r") as f:
        nftlist = f.attrs["number_data_models"]
        ftlist = [
            convert_hdf_to_flagtable(f["FlagTable%d" % i]) for i in range(nftlist)
        ]
        if nftlist == 1:
            return ftlist[0]
        else:
            return ftlist


def convert_gaintable_to_hdf(gt: GainTable, f):
    """Convert a GainTable to an HDF file

    :param gt: GainTable
    :param f: hdf group
    :return: group with gt added
    """
    if not isinstance(gt, xarray.Dataset):
        raise ValueError(f"gt is not an xarray.Dataset: {gt}")
    if gt.attrs["rascil_data_model"] != "GainTable":
        raise ValueError(f"gt is not a GainTable: {GainTable}")

    f.attrs["rascil_data_model"] = "GainTable"
    f.attrs["receptor_frame"] = gt.receptor_frame.type
    f.attrs["phasecentre_coords"] = gt.phasecentre.to_string()
    f.attrs["phasecentre_frame"] = gt.phasecentre.frame.name
    datavars = ["time", "gain", "weight", "residual", "interval", "frequency"]
    for var in datavars:
        f["data_{}".format(var)] = gt[var].data
    return f


def convert_hdf_to_gaintable(f):
    """Convert HDF root to a GainTable

    :param f: hdf group
    :return: GainTable
    """
    assert f.attrs["rascil_data_model"] == "GainTable", "Not a GainTable"
    receptor_frame = ReceptorFrame(f.attrs["receptor_frame"])
    s = f.attrs["phasecentre_coords"].split()
    ss = [float(s[0]), float(s[1])] * u.deg
    phasecentre = SkyCoord(ra=ss[0], dec=ss[1], frame=f.attrs["phasecentre_frame"])

    time = f["data_time"][()]
    frequency = f["data_frequency"][()]
    gain = f["data_gain"][()]
    weight = f["data_weight"][()]
    residual = f["data_residual"][()]
    interval = f["data_interval"][()]
    gt = GainTable(
        time=time,
        frequency=frequency,
        gain=gain,
        weight=weight,
        residual=residual,
        interval=interval,
        receptor_frame=receptor_frame,
        phasecentre=phasecentre,
    )
    return gt


def export_gaintable_to_hdf5(gt: GainTable, filename):
    """Export a GainTable or list to HDF5 format

    :param gt: GainTable or list
    :param filename:
    :return:
    """

    if not isinstance(gt, collections.abc.Iterable):
        gt = [gt]
    with h5py.File(filename, "w") as f:
        if isinstance(gt, list):
            f.attrs["number_data_models"] = len(gt)
            for i, g in enumerate(gt):
                gf = f.create_group("GainTable%d" % i)
                convert_gaintable_to_hdf(g, gf)
        else:
            f.attrs["number_data_models"] = 1
            gf = f.create_group("GainTable%d" % 0)
            convert_gaintable_to_hdf(gt, gf)

        f.flush()


def import_gaintable_from_hdf5(filename):
    """Import GainTable(s) from HDF5 format

    :param filename:
    :return: single gaintable or list of gaintables
    """

    with h5py.File(filename, "r") as f:
        ngtlist = f.attrs["number_data_models"]
        gtlist = [
            convert_hdf_to_gaintable(f["GainTable%d" % i]) for i in range(ngtlist)
        ]
        if ngtlist == 1:
            return gtlist[0]
        else:
            return gtlist


def convert_pointingtable_to_hdf(pt: PointingTable, f):
    """Convert a PointingTable to an HDF file

    :param pt: PointingTable
    :param f: hdf group
    :return: group with pt added
    """
    if not isinstance(pt, xarray.Dataset):
        raise ValueError(f"pt is not an xarray.Dataset: {pt}")
    if pt.attrs["rascil_data_model"] != "PointingTable":
        raise ValueError(f"pt is not a PointingTable: {pt}")

    f.attrs["rascil_data_model"] = "PointingTable"
    f.attrs["receptor_frame"] = pt.receptor_frame.type
    f.attrs["pointingcentre_coords"] = pt.pointingcentre.to_string()
    f.attrs["pointingcentre_frame"] = pt.pointingcentre.frame.name
    f.attrs["pointing_frame"] = pt.pointing_frame
    datavars = [
        "time",
        "nominal",
        "pointing",
        "weight",
        "residual",
        "interval",
        "frequency",
    ]
    for var in datavars:
        f["data_{}".format(var)] = pt[var].data
    f = convert_configuration_to_hdf(pt.configuration, f)
    return f


def convert_hdf_to_pointingtable(f):
    """Convert HDF root to a PointingTable

    :param f: hdf group
    :return: PointingTable
    """
    assert f.attrs["rascil_data_model"] == "PointingTable", "Not a PointingTable"
    receptor_frame = ReceptorFrame(f.attrs["receptor_frame"])
    s = f.attrs["pointingcentre_coords"].split()
    ss = [float(s[0]), float(s[1])] * u.deg
    pointingcentre = SkyCoord(
        ra=ss[0], dec=ss[1], frame=f.attrs["pointingcentre_frame"]
    )
    pointing_frame = f.attrs["pointing_frame"]
    configuration = convert_configuration_from_hdf(f)

    time = f["data_time"][()]
    frequency = f["data_frequency"][()]
    pointing = f["data_pointing"][()]
    nominal = f["data_nominal"][()]
    weight = f["data_weight"][()]
    residual = f["data_residual"][()]
    interval = f["data_interval"][()]

    pt = PointingTable(
        time=time,
        pointing=pointing,
        nominal=nominal,
        weight=weight,
        residual=residual,
        interval=interval,
        frequency=frequency,
        receptor_frame=receptor_frame,
        pointing_frame=pointing_frame,
        pointingcentre=pointingcentre,
        configuration=configuration,
    )
    return pt


def export_pointingtable_to_hdf5(pt: PointingTable, filename):
    """Export a PointingTable or list to HDF5 format

    :param pt:
    :param filename:
    :return:
    """

    if not isinstance(pt, collections.abc.Iterable):
        pt = [pt]
    with h5py.File(filename, "w") as f:
        if isinstance(pt, list):
            f.attrs["number_data_models"] = len(pt)
            for i, v in enumerate(pt):
                vf = f.create_group("PointingTable%d" % i)
                convert_pointingtable_to_hdf(v, vf)
        else:
            f.attrs["number_data_models"] = 1
            vf = f.create_group("PointingTable%d" % 0)
            convert_pointingtable_to_hdf(pt, vf)
        f.flush()


def import_pointingtable_from_hdf5(filename):
    """Import PointingTable(s) from HDF5 format

    :param filename:
    :return: single pointingtable or list of pointingtables
    """

    with h5py.File(filename, "r") as f:
        nptlist = f.attrs["number_data_models"]
        ptlist = [
            convert_hdf_to_pointingtable(f["PointingTable%d" % i])
            for i in range(nptlist)
        ]
        if nptlist == 1:
            return ptlist[0]
        else:
            return ptlist


def convert_skycomponent_to_hdf(sc: Skycomponent, f):
    """Convert Skycomponent to HDF

    :param sc: SkyComponent
    :param f: HDF root
    :return: group with sc added
    """
    if not isinstance(sc, Skycomponent):
        raise ValueError(f"sc is not a SkyComponent: {sc}")

    f.attrs["rascil_data_model"] = "Skycomponent"
    f.attrs["direction"] = convert_direction_to_string(sc.direction)
    f.attrs["frequency"] = sc.frequency
    f.attrs["polarisation_frame"] = sc.polarisation_frame.type
    f.attrs["flux"] = sc.flux
    f.attrs["shape"] = sc.shape
    f.attrs["params"] = str(sc.params)
    f.attrs["name"] = numpy.string_(sc.name)
    return f


def convert_hdf_to_skycomponent(f):
    """Convert HDF root to a SkyComponent

    :param f: hdf group
    :return: SkyComponent
    """
    assert f.attrs["rascil_data_model"] == "Skycomponent", "Not a Skycomponent"
    direction = convert_direction_from_string(f.attrs["direction"])
    frequency = numpy.array(f.attrs["frequency"])
    name = f.attrs["name"]
    polarisation_frame = PolarisationFrame(f.attrs["polarisation_frame"])
    flux = f.attrs["flux"]
    shape = f.attrs["shape"]
    params = ast.literal_eval(f.attrs["params"])
    sc = Skycomponent(
        direction=direction,
        frequency=frequency,
        name=name,
        flux=flux,
        polarisation_frame=polarisation_frame,
        shape=shape,
        params=params,
    )
    return sc


def export_skycomponent_to_hdf5(sc: Union[Skycomponent, list], filename):
    """Export a Skycomponent or list to HDF5 format

    :param sc: SkyComponent
    :param filename:
    :return:
    """

    if not isinstance(sc, collections.abc.Iterable):
        sc = [sc]
    with h5py.File(filename, "w") as f:
        f.attrs["number_data_models"] = len(sc)
        for i, s in enumerate(sc):
            sf = f.create_group("Skycomponent%d" % i)
            convert_skycomponent_to_hdf(s, sf)
        f.flush()


def import_skycomponent_from_hdf5(filename):
    """Import Skycomponent(s) from HDF5 format

    :param filename:
    :return: single skycomponent or list of skycomponents
    """

    with h5py.File(filename, "r") as f:
        nsclist = f.attrs["number_data_models"]
        sclist = [
            convert_hdf_to_skycomponent(f["Skycomponent%d" % i]) for i in range(nsclist)
        ]
        if nsclist == 1:
            return sclist[0]
        else:
            return sclist


def convert_image_to_hdf(im: Image, f):
    """Convert an Image to an HDF file

    :param im: Image
    :param f: hdf group
    :return: group with im added
    """
    if not isinstance(im, xarray.Dataset):
        raise ValueError(f"im is not xarray dataset {im}")
    if im.attrs["rascil_data_model"] != "Image":
        raise ValueError(f"fim is not an Image: {im}")

    f.attrs["rascil_data_model"] = "Image"
    f["data"] = im["pixels"].data
    f.attrs["wcs"] = numpy.string_(im.image_acc.wcs.to_header_string())
    f.attrs["phasecentre_coords"] = im.image_acc.phasecentre.to_string()
    f.attrs["phasecentre_frame"] = im.image_acc.phasecentre.frame.name
    f.attrs["polarisation_frame"] = im.image_acc.polarisation_frame.type
    f.attrs["frequency"] = im.frequency

    return f


def convert_hdf_to_image(f):
    """Convert HDF root to an Image

    :param f: hdf group
    :return: Image
    """
    if (
        "rascil_data_model" in f.attrs.keys()
        and f.attrs["rascil_data_model"] == "Image"
    ):
        polarisation_frame = PolarisationFrame(f.attrs["polarisation_frame"])
        wcs = WCS(f.attrs["wcs"])
        frequency = numpy.array(f.attrs["frequency"])
        data = numpy.array(f["data"])
        s = f.attrs["phasecentre_coords"].split()
        ss = [float(s[0]), float(s[1])] * u.deg
        phasecentre = SkyCoord(ra=ss[0], dec=ss[1], frame=f.attrs["phasecentre_frame"])
        im = Image(data=data, polarisation_frame=polarisation_frame, wcs=wcs)
        return im
    else:
        return None


def export_image_to_hdf5(im, filename):
    """Export an Image or list to HDF5 format

    :param im:
    :param filename:
    :return:
    """

    if not isinstance(im, collections.abc.Iterable):
        im = [im]
    with h5py.File(filename, "w") as f:
        if isinstance(im, list):
            f.attrs["number_data_models"] = len(im)
            for i, v in enumerate(im):
                vf = f.create_group("Image%d" % i)
                convert_image_to_hdf(v, vf)
        else:
            f.attrs["number_data_models"] = 1
            vf = f.create_group("Image%d" % 0)
            convert_image_to_hdf(im, vf)
        f.flush()
        f.close()


def import_image_from_hdf5(filename):
    """Import Image(s) from HDF5 format

    :param filename:
    :return: single image or list of images
    """

    with h5py.File(filename, "r") as f:
        nimlist = f.attrs["number_data_models"]
        imlist = [convert_hdf_to_image(f["Image%d" % i]) for i in range(nimlist)]
        if nimlist == 1:
            return imlist[0]
        else:
            return imlist


def export_skymodel_to_hdf5(sm, filename):
    """Export a Skymodel or list to HDF5 format

    :param sm: SkyModel or list of SkyModels
    :param filename:
    """

    if not isinstance(sm, collections.abc.Iterable):
        sm = [sm]

    with h5py.File(filename, "w") as f:
        f.attrs["number_data_models"] = len(sm)
        for i, s in enumerate(sm):
            sf = f.create_group("SkyModel%d" % i)
            convert_skymodel_to_hdf(s, sf)
        f.flush()
        f.close()


def convert_skymodel_to_hdf(sm, f):
    """Convert a SkyModel to an HDF file

    :param sm: Skymodel
    :param f: hdf group
    :return: group with skymodel added
    """
    if not isinstance(sm, SkyModel):
        raise ValueError(f"sm is not a SkyModel: {sm}")

    f.attrs["rascil_data_model"] = "SkyModel"
    f.attrs["fixed"] = sm.fixed
    if sm.components is not None:
        f.attrs["number_skycomponents"] = len(sm.components)
        for i, sc in enumerate(sm.components):
            cf = f.create_group("skycomponent%d" % i)
            convert_skycomponent_to_hdf(sm.components[i], cf)
        else:
            f.attrs["number_skycomponents"] = len(sm.components)
    if sm.image is not None:
        cf = f.create_group("image")
        convert_image_to_hdf(sm.image, cf)
    if sm.mask is not None:
        cf = f.create_group("mask")
        convert_image_to_hdf(sm.mask, cf)
    if sm.gaintable is not None:
        cf = f.create_group("gaintable")
        convert_gaintable_to_hdf(sm.gaintable, cf)
    return f


def import_skymodel_from_hdf5(filename):
    """Import a Skymodel or list from HDF5 format

    :param filename:
    :return: SkyModel
    """

    with h5py.File(filename, "r") as f:
        nsmlist = f.attrs["number_data_models"]
        smlist = [convert_hdf_to_skymodel(f["SkyModel%d" % i]) for i in range(nsmlist)]
        if nsmlist == 1:
            return smlist[0]
        else:
            return smlist


def convert_hdf_to_skymodel(f):
    """Convert HDF to SkyModel

    :param f: hdf group
    :return: SkyModel
    """
    assert f.attrs["rascil_data_model"] == "SkyModel", f.attrs["rascil_data_model"]

    fixed = f.attrs["fixed"]

    ncomponents = f.attrs["number_skycomponents"]
    components = list()
    for i in range(ncomponents):
        cf = f[("skycomponent%d" % i)]
        components.append(convert_hdf_to_skycomponent(cf))
    if "image" in f.keys():
        cf = f["image"]
        image = convert_hdf_to_image(cf)
    else:
        image = None
    if "mask" in f.keys():
        cf = f["mask"]
        mask = convert_hdf_to_image(cf)
    else:
        mask = None
    if "gaintable" in f.keys():
        cf = f["gaintable"]
        gaintable = convert_hdf_to_gaintable(cf)
    else:
        gaintable = None

    return SkyModel(
        image=image, components=components, gaintable=gaintable, mask=mask, fixed=fixed
    )


def convert_griddata_to_hdf(gd: GridData, f):
    """Convert a GridDatato an HDF file

    :param gd: GridData
    :param f: hdf group
    :return: group with gd added
    """
    if not isinstance(gd, xarray.Dataset):
        raise ValueError(f"gd is not an xarray.Dataset: {gd}")
    if gd.attrs["rascil_data_model"] != "GridData":
        raise ValueError(f"gd is not a GridData: {gd}")

    f.attrs["rascil_data_model"] = "GridData"
    f["data"] = gd["pixels"].data

    f.attrs["grid_wcs"] = numpy.string_(gd.griddata_acc.griddata_wcs.to_header_string())
    f.attrs["polarisation_frame"] = gd.griddata_acc.polarisation_frame.type
    return f


def convert_hdf_to_griddata(f):
    """Convert HDF root to a GridData

    :param f: hdf group
    :return: GridData
    """
    assert f.attrs["rascil_data_model"] == "GridData", "Not a GridData"
    data = numpy.array(f["data"])
    grid_wcs = WCS(f.attrs["grid_wcs"])
    polarisation_frame = PolarisationFrame(f.attrs["polarisation_frame"])

    gd = GridData(data=data, polarisation_frame=polarisation_frame, grid_wcs=grid_wcs)

    return gd


def export_griddata_to_hdf5(gd, filename):
    """Export a GridData or list to HDF5 format

    :param gd:
    :param filename:
    :return:
    """

    if not isinstance(gd, collections.abc.Iterable):
        gd = [gd]
    with h5py.File(filename, "w") as f:
        if isinstance(gd, list):
            f.attrs["number_data_models"] = len(gd)
            for i, v in enumerate(gd):
                vf = f.create_group("GridData%d" % i)
                convert_image_to_hdf(v, vf)
        else:
            f.attrs["number_data_models"] = 1
            vf = f.create_group("GridData%d" % 0)
            convert_griddata_to_hdf(gd, vf)
        f.flush()
        f.close()


def import_griddata_from_hdf5(filename):
    """Import GridData(s) from HDF5 format

    :param filename:
    :return: single image or list of images
    """

    with h5py.File(filename, "r") as f:
        nimlist = f.attrs["number_data_models"]
        gdlist = [convert_hdf_to_griddata(f["GridData%d" % i]) for i in range(nimlist)]
        if nimlist == 1:
            return gdlist[0]
        else:
            return gdlist


def convert_convolutionfunction_to_hdf(cf: ConvolutionFunction, f):
    """Convert a ConvolutionFunction to an HDF file

    :param cf: ConvolutionFunction
    :param f: hdf group
    :return: group with cf added
    """
    if not isinstance(cf, xarray.Dataset):
        raise ValueError("cf is not an xarray.Dataset")
    if cf.attrs["rascil_data_model"] != "ConvolutionFunction":
        raise ValueError(f"cf is not a ConvolutionFunction: {cf}")

    f.attrs["rascil_data_model"] = "ConvolutionFunction"
    f["data"] = cf["pixels"].data
    f.attrs["grid_wcs"] = numpy.string_(
        cf.convolutionfunction_acc.cf_wcs.to_header_string()
    )
    f.attrs["polarisation_frame"] = cf.convolutionfunction_acc.polarisation_frame.type
    return f


def convert_hdf_to_convolutionfunction(f):
    """Convert HDF root to a ConvolutionFunction

    :param f: hdf group
    :return: ConvolutionFunction
    """
    assert f.attrs["rascil_data_model"] == "ConvolutionFunction", f.attrs[
        "rascil_data_model"
    ]
    data = numpy.array(f["data"])
    polarisation_frame = PolarisationFrame(f.attrs["polarisation_frame"])
    cf_wcs = WCS(f.attrs["grid_wcs"])
    gd = ConvolutionFunction(data, cf_wcs=cf_wcs, polarisation_frame=polarisation_frame)
    return gd


def export_convolutionfunction_to_hdf5(cf, filename):
    """Export a ConvolutionFunction to HDF5 format

    :param cf:
    :param filename:
    :return:
    """

    if not isinstance(cf, collections.abc.Iterable):
        cf = [cf]
    with h5py.File(filename, "w") as f:
        if isinstance(cf, list):
            f.attrs["number_data_models"] = len(cf)
            for i, v in enumerate(cf):
                vf = f.create_group("ConvolutionFunction%d" % i)
                convert_convolutionfunction_to_hdf(v, vf)
        else:
            f.attrs["number_data_models"] = 1
            vf = f.create_group("ConvolutionFunction%d" % 0)
            convert_convolutionfunction_to_hdf(cf, vf)
        f.flush()
        f.close()


def import_convolutionfunction_from_hdf5(filename):
    """Import ConvolutionFunction(s) from HDF5 format

    :param filename:
    :return: single image or list of images
    """

    with h5py.File(filename, "r") as f:
        nimlist = f.attrs["number_data_models"]
        cflist = [
            convert_hdf_to_convolutionfunction(f["ConvolutionFunction%d" % i])
            for i in range(nimlist)
        ]
        if nimlist == 1:
            return cflist[0]
        else:
            return cflist


def memory_data_model_to_buffer(model, jbuff, dm):
    """Copy a memory data model to a buffer data model

    The file type is derived from the file extension. All are hdf only.

    :param model: Memory data model to be sent to buffer
    :param jbuff: JSON describing buffer
    :param dm: JSON describing data model
    """
    name = jbuff["directory"] + dm["name"]

    import os

    _, file_extension = os.path.splitext(dm["name"])

    if dm["data_model"] == "BlockVisibility":
        return export_blockvisibility_to_hdf5(model, name)
    elif dm["data_model"] == "Image":
        return export_image_to_hdf5(model, name)
    elif dm["data_model"] == "GridData":
        return export_griddata_to_hdf5(model, name)
    elif dm["data_model"] == "ConvolutionFunction":
        return export_convolutionfunction_to_hdf5(model, name)
    elif dm["data_model"] == "SkyModel":
        return export_skymodel_to_hdf5(model, name)
    elif dm["data_model"] == "GainTable":
        return export_gaintable_to_hdf5(model, name)
    elif dm["data_model"] == "FlagTable":
        return export_flagtable_to_hdf5(model, name)
    elif dm["data_model"] == "PointingTable":
        return export_pointingtable_to_hdf5(model, name)
    else:
        raise ValueError("Data model %s not supported" % dm["data_model"])


def buffer_data_model_to_memory(jbuff, dm):
    """Copy a buffer data model into memory data model

    The file type is derived from the file extension. All are hdf only.

    :param jbuff: JSON describing buffer
    :param dm: JSON describing data model
    :return: data model
    """
    import os

    name = os.path.join(jbuff["directory"], dm["name"])

    import os

    _, file_extension = os.path.splitext(dm["name"])

    if dm["data_model"] == "BlockVisibility":
        return import_blockvisibility_from_hdf5(name)
    elif dm["data_model"] == "Image":
        return import_image_from_hdf5(name)
    elif dm["data_model"] == "SkyModel":
        return import_skymodel_from_hdf5(name)
    elif dm["data_model"] == "GainTable":
        return import_gaintable_from_hdf5(name)
    elif dm["data_model"] == "FlagTable":
        return import_flagtable_from_hdf5(name)
    elif dm["data_model"] == "PointingTable":
        return import_pointingtable_from_hdf5(name)
    elif dm["data_model"] == "GridData":
        return import_griddata_from_hdf5(name)
    elif dm["data_model"] == "ConvolutionFunction":
        return import_convolutionfunction_from_hdf5(name)
    else:
        raise ValueError("Data model %s not supported" % dm["data_model"])


def data_model_equals(ds_new, ds_ref, verbose=False):
    """

    :param ds_ref:
    :param ds_new:
    :return:
    """
    equal = True
    for coord in ds_ref.coords:
        if not ds_new[coord].equals(ds_ref[coord]):
            equal = False
            log.warning("data_model_equals: Coordinate {} differs".format(coord))
            if verbose:
                log.warning(
                    "data_model_equals: New {}\ndata_model_equals: Reference {}".format(
                        ds_new[coord], ds_ref[coord]
                    )
                )
    if equal:
        for var in ds_ref.data_vars:
            if not ds_new[var].equals(ds_ref[var]):
                equal = False
                log.warning("data_model_equals: Data variable {} differs".format(var))
                if verbose:
                    log.warning(
                        "data_model_equals: New {}\ndata_model_equals: Reference {}".format(
                            ds_new[var], ds_ref[var]
                        )
                    )
    if equal:
        for attr in ds_ref.attrs.keys():
            if not attr in ds_new.attrs.keys():
                equal = False
                log.warning("Attribute {} missing in ds_new".format(attr))
            # The attribute may not have a comparison operator
            try:
                if not ds_ref.attrs[attr] == ds_new.attrs[attr]:
                    log.warning("data_model_equals: Attribute {} differs".format(attr))
            except:
                log.warning(
                    "data_model_equals: Attribute {} cannot be compared".format(attr)
                )

    return equal and ds_ref.equals(ds_new)
