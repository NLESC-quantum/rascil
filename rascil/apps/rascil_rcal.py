""" RASCIL MS advice

"""

import argparse
import logging
import os
import pprint
import sys
from typing import Iterable

import matplotlib
import numpy
from ska_sdp_func import sum_threshold_rfi_flagger

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.time import Time
from astropy.visualization import time_support
from astropy.coordinates import SkyCoord

from rascil.data_models import (
    BlockVisibility,
    GainTable,
    PolarisationFrame,
    export_gaintable_to_hdf5,
    import_skycomponent_from_hdf5,
)
from rascil.processing_components import (
    create_blockvisibility_from_ms,
    solve_gaintable,
    dft_skycomponent_visibility,
    copy_visibility,
    copy_gaintable,
    create_skycomponent,
    create_gaintable_from_blockvisibility,
    concatenate_gaintables,
    create_low_test_beam,
    create_image_from_visibility,
    apply_beam_to_skycomponent,
)
from rascil.processing_components.util.coordinate_support import hadec_to_azel
from rascil.processing_components.image.operations import (
    import_image_from_fits,
    export_image_to_fits,
)

log = logging.getLogger("rascil-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


class FileFormatError(Exception):
    pass


def cli_parser():
    """Get a command line parser and populate it with arguments

    First a CLI argument parser is created. Each function call adds more arguments to the parser.

    :return: CLI parser argparse
    """

    parser = argparse.ArgumentParser(
        description="RASCIL RCAL simulator", fromfile_prefix_chars="@"
    )
    parser.add_argument(
        "--ingest_msname", type=str, default=None, help="MeasurementSet to be read"
    )
    parser.add_argument(
        "--ingest_dd",
        type=int,
        nargs="*",
        default=[0],
        help="Data descriptors in MS to read (all must have the same number of channels)",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default=None,
        help="Name of logfile (default is to construct one from msname)",
    )

    parser.add_argument(
        "--ingest_components_file",
        type=str,
        default=None,
        help="Name of components file (HDF5/txt) format",
    )
    parser.add_argument(
        "--apply_beam",
        type=str,
        default=False,
        help="If yes, apply primary beam correction to the ingested components",
    )
    parser.add_argument(
        "--ingest_beam_file",
        type=str,
        default=None,
        help="Name of external beam file in FITS format",
    )

    parser.add_argument(
        "--do_plotting",
        type=str,
        default=False,
        help="If yes, plot the gain table values over time",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=None,
        help="Full path of the directory to save the gain plots into "
        "(default is the same directory the MS file is located)",
    )
    parser.add_argument(
        "--plot_dynamic",
        type=str,
        default=False,
        help="If yes, enable dynamic plotting and save animation; if no, just save the final gain table",
    )
    parser.add_argument(
        "--use_previous_gaintable",
        type=str,
        default="False",
        help="Use previous gaintable as starting point for solution",
    )

    parser.add_argument(
        "--phase_only_solution",
        type=str,
        default="True",
        help="Solution should be for phases only",
    )

    parser.add_argument(
        "--solution_tolerance",
        type=float,
        default=1e-12,
        help="Tolerance for solution: stops iteration when changes below this level",
    )

    parser.add_argument(
        "--flag_rfi",
        type=str,
        default="False",
        help="Whether to run the RFI flagger (before obtaining calibration solutions), or not.",
    )

    parser.add_argument(
        "--initial_threshold",
        type=float,
        default=8.0,
        help="The initial threshold to be used by the flagger. "
        "Used for calculating a list of thresholds."
        "Note: use default value since flagger is still under development",
    )

    parser.add_argument(
        "--rho",
        type=float,
        default=1.5,
        help="The initial rho used by flagger. Used for calculating a list of thresholds. "
        "Note: use default value since flagger is still under development",
    )

    return parser


def _rfi_flagger(bvis, initial_threshold=8, rho=1.5):
    """
    Wrapper function for the SKA flagger, certain defaults are managed here.
    Versions of flagger:
    1. (https://gitlab.com/ska-telescope/ska-post-correlation-rfi-flagger) (deprecated)
    2. (https://gitlab.com/ska-telescope/sdp/ska-sdp-func/-/blob/main/src/ska_sdp_func/rfi_flagger.py) (current)

    The code provides a sequence and derives the best experimented thresholds for flagging.
    For a longer sequence, the flagging threshold should be lower.
    Details see line 158, Offringa et al. 2010MNRAS.405..155O
    The parameters have been fixed for SKA purposes.

    :param bvis: Block visibility
    :param initial_threshold: The initial threshold to be used
    :param rho: Empirical parameter to derive the thresholds to be used
    :return: Block visibility with flags populated.
    """

    # Set up the sequence.
    sequence = numpy.array([1, 2, 4, 8, 16, 32], dtype=numpy.int32)
    thresholds = initial_threshold / numpy.power(rho, numpy.log2(sequence))
    max_sequence_length = 2 ** (len(thresholds) - 1)

    vis_data = bvis["vis"].data
    flag_data = bvis["flags"].data.astype(numpy.int32)

    log.info("The dimensions of the visibility data:{}".format(vis_data.shape))

    sum_threshold_rfi_flagger(vis_data, thresholds, flag_data, max_sequence_length)

    # update flag data in place
    bvis["flags"].data = flag_data


def rcal_simulator(bvis, args):
    """RCAL simulator

    Generate real-time calibration tables and optionally apply gains to data.
    RFI flagging also takes place as part of the pipeline (currently only place-holder for real function).

    :param bvis: a single BlockVisibility object with one or more time samples
    :param args: argparse with appropriate arguments
    :return:
        gtfile: file name containing all of the GainTables
    """
    log.info(pprint.pformat(vars(args)))
    log.info("Processing bvis with dimensions: %s", bvis.dims)

    if args.flag_rfi == "True":
        _rfi_flagger(bvis, args.initial_threshold, args.rho)

    if args.ingest_components_file is not None:
        log.info("Using components model for calibration")

        try:
            model_components = import_skycomponent_from_hdf5(
                args.ingest_components_file
            )
            telescope_name = bvis.configuration.name
            log.info(f"Read HDF components file {args.ingest_components_file}")

        except OSError:
            # file is not HDF-compatible, trying txt
            if ".txt" in args.ingest_components_file:

                # Use the polarisation from the BlockVisibility
                pol = PolarisationFrame(bvis._polarisation_frame)
                log.info(f"Use Polarisation Frame {pol.names}")

                model_components = read_skycomponent_from_txt_with_external_frequency(
                    args.ingest_components_file, bvis.frequency, pol
                )
                log.info(f"Read text components file {args.ingest_components_file}")
                telescope_name = "LOW"  # Since LOW doesn't write configuration, set arbitrarily for now

            else:
                raise FileFormatError("Input file must be of format: hdf or txt.")

        if args.apply_beam == "True":
            log.info("Apply beam correction to the components")
            model_components = apply_beam_correction(
                bvis,
                model_components,
                args.ingest_beam_file,
                telescope_name=telescope_name,
            )
    else:
        log.info(f"Using point source model for calibration")
        model_components = None

    if args.plot_dir is None:
        plot_dir = os.getcwd()
    else:
        plot_dir = args.plot_dir

    bvis_gen = bvis_source(bvis)

    gt_gen = bvis_solver(
        bvis_gen,
        model_components,
        phase_only=args.phase_only_solution == "True",
        use_previous=args.use_previous_gaintable == "True",
        tol=args.solution_tolerance,
    )

    datetime = (
        bvis.datetime.data[0].__str__().split(".")[0].replace(":", "").replace("-", "")
    )
    base = os.path.basename(args.ingest_msname)
    plotfile = plot_dir + "/" + base.replace(".ms", f"_{datetime}_plot")
    log.info(f"Write plots into : \n{plotfile}\n")

    do_plotting = args.do_plotting == "True"
    plot_dynamic = args.plot_dynamic == "True"
    full_gt = gt_sink(gt_gen, do_plotting, plot_dynamic, plotfile)
    log.info(args.ingest_msname)
    gtfile = args.ingest_msname.replace(".ms", f"_{datetime}_gaintable.hdf")
    export_gaintable_to_hdf5(full_gt, gtfile)

    return gtfile


def bvis_source(bvis: BlockVisibility, dim="time") -> Iterable[BlockVisibility]:
    """Creates a BlockVisibility generator that iterates in the specified dimension

    :param bvis: BlockVisibility
    :param dim: dimension to be iterated
    :return: generator of BlockVisibility
    """
    log.info("\nrascil_rcal_bvis_source: BlockVisibility source starting")
    for time, bv in bvis.groupby(dim, squeeze=False):
        yield bv
    log.info("rascil_rcal_bvis_source: BlockVisibility source has finished\n")


def bvis_solver(
    bvis_gen: Iterable[BlockVisibility],
    model_components,
    phase_only=False,
    use_previous=True,
    jones_type="B",
    tol=1e-6,
) -> Iterable[GainTable]:
    """Iterate through the block vis, solving for the gain, returning gaintable generator

    Optionally takes a list of skycomponents to use as a model
    Optionally, apply calibration to input BlockVisibilities (done in place)

    :param bvis_gen: Generator of BlockVisibility
    :param model_components: Model components
    :param phase_only: Solve for phase only? Otherwise, also solve for amplitude
    :param use_previous: if True, use previous GainTable as starting point for solution
    :param jones_type: Type of calibration matrix T or G or B
    :param tol: solution tolerance
    :return: generator of GainTables
    """
    previous = None
    for bv in bvis_gen:
        gt, previous = realtime_single_bvis_solver(
            bv,
            model_components,
            previous,
            phase_only=phase_only,
            jones_type=jones_type,
            tol=tol,
            use_previous=use_previous,
        )
        yield gt


def realtime_single_bvis_solver(
    bvis: BlockVisibility,
    model_components,
    previous_solution,
    phase_only=False,
    jones_type="B",
    tol=1e-6,
    use_previous=True,
):
    """
    The bulk of running RCAL.
    This solves a single BlockVisibility and returns its GainTable.

    :param bvis: a BlockVisibility object
    :param model_components: Model components
    :param previous_solution: previous GainTable used as starting point;
                              if use_previous is True, update this object in place
                              and make it the "previous" to be used next time
    :param phase_only: Solve for phase only? Otherwise, also solve for amplitude
    :param jones_type: Type of calibration matrix T or G or B
    :param tol: solution tolerance
    :param use_previous: if True, use previous GainTable as starting point for solution

    :return: GainTable for the input BlockVisibility
    """
    if model_components is not None:
        modelvis = copy_visibility(bvis, zero=True)
        modelvis = dft_skycomponent_visibility(modelvis, model_components)
        gt = solve_gaintable(
            bvis,
            modelvis=modelvis,
            gt=previous_solution,
            phase_only=phase_only,
            jones_type=jones_type,
            tol=tol,
        )
    else:
        gt = solve_gaintable(
            bvis,
            gt=previous_solution,
            phase_only=phase_only,
            jones_type=jones_type,
            tol=tol,
        )

    if use_previous:
        newgt = create_gaintable_from_blockvisibility(bvis, jones_type=jones_type)
        previous_solution = copy_gaintable(gt)
        previous_solution = previous_solution.assign_coords(
            {"time": newgt["time"].data}
        )

    return gt, previous_solution


def gt_sink(gt_gen: Iterable[GainTable], do_plotting, plot_dynamic, plot_name):
    """Iterate through the gaintables, logging resisual, combine into single GainTable

    :param gt_gen: Generator of GainTables
    :param do_plotting: Option to plot gain tables
    :param plot_dynamic: Option to make it a dynamic plot
    :param plot_name: File name for the plot (contains directory name)

    :return: GainTable
    """
    gt_list = list()
    for gt in gt_gen:
        datetime = gt["datetime"][0].data
        log.info(
            f"rascil_rcal_gt_sink: Processing integration {datetime} "
            f"residual: {numpy.max(gt.residual.data):.3g}"
        )
        gt_list.append(gt)

        # Tentative dynamic plotting routine
    #        if do_plotting and plot_dynamic and len(gt_list) > 1:
    #            dynamic_update(gt_list, plot_name)
    #            log.info(f"Done dynamic plotting for {datetime}.")

    if do_plotting:
        gt_single_plot(gt_list, plot_name)
        log.info("Save final plot.")

    full_gt = concatenate_gaintables(gt_list, "time")
    return full_gt


def get_gain_data(gt_list):
    """Get data from a list of GainTables used for plotting.

    :param gt_list: GainTable list to plot

    :return: List of arrays in format of [time, amplitude-1, phase-phase(antenna0), residual]

    """

    def angle_wrap(angle):

        if angle > 180.0:
            angle = 360.0 - angle
        if angle < -180.0:
            angle = 360.0 + angle

        return angle

    if not isinstance(gt_list, list):
        gt_list = [gt_list]

    with time_support(format="iso", scale="utc"):

        gains = []
        residual = []
        time = []
        weight = []

        # We only look at the central channel at the moment
        half_of_chans_to_avg = 0
        for gt in gt_list:
            time.append(gt.time.data[0] / 86400.0)
            current_gain = gt.gain.data[0]
            nchan = current_gain.shape[1]
            central_chan = nchan // 2
            gains.append(
                numpy.average(
                    current_gain[
                        :,
                        central_chan
                        - half_of_chans_to_avg : central_chan
                        + half_of_chans_to_avg
                        + 1,
                        0,
                        0,
                    ],
                    axis=1,
                )
            )
            residual.append(
                numpy.average(
                    gt.residual.data[
                        0,
                        central_chan
                        - half_of_chans_to_avg : central_chan
                        + half_of_chans_to_avg
                        + 1,
                        0,
                        0,
                    ],
                    axis=0,
                )
            )
            weight.append(
                numpy.average(
                    gt.weight.data[
                        0,
                        :,
                        central_chan
                        - half_of_chans_to_avg : central_chan
                        + half_of_chans_to_avg
                        + 1,
                        0,
                        0,
                    ],
                    axis=1,
                )
            )

        gains = numpy.array(gains)
        amp = numpy.abs(gains)
        amp = amp.reshape(amp.shape[1], amp.shape[0])
        phase = numpy.angle(gains, deg=True)
        weight = numpy.array(weight)
        weight = weight.reshape(weight.shape[1], weight.shape[0])

        phase_rel = []
        for i in range(len(phase[0])):
            phase_now = phase[:, i] - phase[:, 0]
            phase_now = [angle_wrap(element) for element in phase_now]
            phase_rel.append(phase_now)
        phase_rel = numpy.array(phase_rel)

        timeseries = Time(time, format="mjd", out_subfmt="str")

        return timeseries, amp, phase_rel, residual, weight


def gt_single_plot(gt_list, plot_name=None):
    """Plot gaintable (gain and residual values) over time
       Used to generate a single plot only

    :param gt_list: GainTable list to plot
    :param plot_name: File name for the plot (contains directory name)

    :return
    """

    if not isinstance(gt_list, list):
        gt_list = [gt_list]

    with time_support(format="iso", scale="utc"):
        timeseries, amp, phase_rel, residual, weight = get_gain_data(gt_list)

        plt.cla()
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
        fig.subplots_adjust(hspace=0)

        datetime = gt_list[0]["datetime"][0].data

        for i in range(amp.shape[0]):
            ax1.plot(timeseries, amp[i] - 1, "-", label=f"Antenna {i}")
            ax2.plot(timeseries, phase_rel[i], "-", label=f"Antenna {i}")
            ax3.plot(timeseries, weight[i], "-", label=f"Antenna {i}")

        ax1.ticklabel_format(axis="y", style="scientific", useMathText=True)

        ax1.set_ylabel("Gain Amplitude - 1")
        ax2.set_ylabel("Gain Phase (Antenna - Antenna 0)")
        ax3.set_ylabel("Gain Weight")
        ax3.legend(loc="best")

        ax4.plot(timeseries, residual, "-")
        ax4.set_ylabel("Residual")
        ax4.set_xlabel("Time (UTC)")
        ax4.set_yscale("log")
        plt.xticks(rotation=30)

        fig.suptitle(f"Updated GainTable at {datetime}")
        plt.savefig(plot_name + ".png")

    return


def read_skycomponent_from_txt_with_external_frequency(filename, freq, pol):
    """
    Read source input from a txt file and make them into skycomponents

    :param filename: Name of input file
    :param freq: External frequency data
    :param pol: Polarization frame
    :return comp: List of skycomponents
    """

    nchan = len(freq)
    npol = pol.npol
    log.info(f" nchan = {nchan}, npol = {npol}")

    # The txt file needs to have the first three columns in the format of (RA, Dec, flux(stokesI))
    # We currently only read in stokesI components
    # TODO: read in the full polarisation
    data = numpy.loadtxt(filename, delimiter=",", unpack=True)
    ra = data[0]
    dec = data[1]
    flux = data[2]

    # Single element.
    if numpy.isscalar(ra):
        direc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs", equinox="J2000")
        flux_array = numpy.zeros((nchan, npol))
        flux_array[:, 0] = flux
        comp = create_skycomponent(
            direction=direc,
            flux=flux_array,
            frequency=freq,
            polarisation_frame=pol,
        )
    else:
        comp = []
        for i, row in enumerate(ra):
            direc = SkyCoord(
                ra=ra[i] * u.deg, dec=dec[i] * u.deg, frame="icrs", equinox="J2000"
            )

            # Temporary: Currently doesn't do frequency correction for flux
            # This should be fixed.
            flux_array = numpy.zeros((nchan, npol))
            flux_array[:, 0] = flux[i]

            comp.append(
                create_skycomponent(
                    direction=direc,
                    flux=flux_array,
                    frequency=freq,
                    polarisation_frame=pol,
                )
            )

    return comp


def apply_beam_correction(
    bvis, components, beam_file, telescope_name=None, write_beam=False
):
    """Apply primary beam to skycomponents for a better skymodel

    :param bvis: Blockvisibility for creating the test beam
    :param components: Input list of skycomponents
    :param beam_file: External FITS file of beam information (regardless of telescope)
    :param telescope_name: Which telescope (MID or LOW)
    :param write_beam: Write the beam?

    :return comp_new: Corrected list of components

    """

    if beam_file is not None:

        log.info("Use external beam image for correction.")
        beam = import_image_from_fits(beam_file)
        comp_new = apply_beam_to_skycomponent(components, beam, inverse=False)

    else:
        if "MID" not in telescope_name and "LOW" not in telescope_name:
            raise ValueError(
                "Telescope configuration is neither for SKA Mid nor for SKA Low."
                "Please specify correct configuration."
            )

        # Currently don't do anything for MID components
        # TODO: Generate primary beam for MID
        if "MID" in telescope_name:
            comp_new = components

        elif "LOW" in telescope_name:

            # The latitude for LOW is -27 degrees
            phasecentre = bvis.phasecentre
            # We want to make the beam for transit so we sett the HA to 0.0
            az, el = hadec_to_azel(0.0 * u.deg, phasecentre.dec, -27.0 * u.deg)
            log.info(f"The azimuth and elevation are: {az.to(u.deg), el.to(u.deg)}")

            # Create a mock image, use the default npixel and cellsize
            npixel = 512
            fov_rad = 16.0 * numpy.pi / 180.0
            cellsize = numpy.arcsin(2.0 * numpy.sin(0.5 * fov_rad) / npixel)
            # cellsize = 16.0 * numpy.pi / (npixel * 180.0)
            model = create_image_from_visibility(
                bvis, cellsize=cellsize, override_cellsize=False
            )
            beam_local = create_low_test_beam(model, use_local=True, azel=(az, el))
            beam = create_low_test_beam(model, use_local=False)
            beam["pixels"].data = beam_local["pixels"].data

            # Output beam image itself for checking purposes
            if write_beam:
                export_image_to_fits(beam, "rascil_low_beam.fits")

            # Check the polarisation match
            # TODO: If they don't, try to use a different approach
            comp_pol = components[0].polarisation_frame
            log.info(
                f"The beam's polarisation frame is {beam.image_acc.polarisation_frame}, and the skycomponents {comp_pol}"
            )

            comp_new = apply_beam_to_skycomponent(components, beam, inverse=False)

    return comp_new


def main(args):
    assert args.ingest_msname is not None, "Input msname must be specified"

    if args.logfile is None:
        logfile = args.ingest_msname.replace(".ms", "_rcal.log")
    else:
        logfile = args.logfile

    def init_logging():
        logging.basicConfig(
            filename=logfile,
            filemode="a",
            format="%(asctime)s.%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%d/%m/%Y %I:%M:%S %p",
            level=logging.INFO,
        )

    init_logging()

    log.info("\nRASCIL RCAL simulator\n")

    log.info(pprint.pformat(vars(args)))

    bvis = create_blockvisibility_from_ms(
        args.ingest_msname, selected_dds=args.ingest_dd
    )[0]
    telescope_name = bvis.configuration.name
    log.info(f"The data is from {telescope_name}.")
    log.info(f"\nMS loaded into BlockVisibility:\n{bvis}\n")

    rcal_simulator(bvis, args)


if __name__ == "__main__":
    parser = cli_parser()
    argv = parser.parse_args()
    main(argv)
