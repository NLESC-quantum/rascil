import os
import logging

import matplotlib.pyplot as plt
import numpy as np
from casacore.tables import table

log = logging.getLogger("rascil-logger")

def get_directory_size(directory):
    """Returns the `directory` size in bytes."""
    total = 0
    try:
        for entry in os.scandir(directory):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_directory_size(entry.path)
    except NotADirectoryError:
        # if `directory` isn't a directory, get the file size then
        log.warning("Not a directory {}".format(directory))
        return os.path.getsize(directory)
    except PermissionError:
        log.warning("Permission problem {}".format(directory))
        # if for whatever reason we can't open the folder, return 0
        return 0
    return total


def display_ms_as_image(msname, nch=None):
    log.info(f"Plotting vis plot for the channel {nch} of the MS {msname}")
    t = table(msname, readonly=True)
    # Ignore autocorrelations
    t = t.query("ANTENNA1 != ANTENNA2")
    ant1 = t.getcol("ANTENNA1")
    ant2 = t.getcol("ANTENNA2")
    timestamps = t.getcol("TIME")

    ms = np.array(t.getcol("DATA"), dtype=np.complex128)  # [:, ch, corr]
    # Flag if one correlation is flagged
    flags = np.any(
        np.array(t.getcol("FLAG"), np.bool), axis=2
    )  # [:, ch] - all frequency should be flagged out
    t.close()
    log.info("# Rows: {}".format(ms.shape[0]))
    log.info("# Channels: {}".format(ms.shape[1]))
    log.info("# Correlations: {}".format(ms.shape[2]))
    log.info("{} % flagged".format(np.sum(flags) / flags.size * 100))
    if nch is None:
        nch = ms.shape[1] // 2

    t = table(os.path.join(msname, "SPECTRAL_WINDOW"), readonly=True)
    freq = t.getcol("CHAN_FREQ")[0]
    if nch < 0:
        nch = 0
    elif nch > (freq.shape[0] - 1):
        nch = freq.shape[0] - 1
    log.info(
        f"MS contains {freq.shape[0]} frequency channel(s), from 0 to {(freq.shape[0] - 1)}"
    )
    log.info(f"Plotting vis for the channel {nch}, freq = {freq[nch]} Hz")
    title_part2 = str(freq[nch]) + " Hz"
    t.close()

    t = table(os.path.join(msname, "ANTENNA"), readonly=True)
    ant_position = np.array(t.getcol("POSITION"))
    t.close()

    baselength = np.zeros(ms.shape[0])
    for i in range(ms.shape[0]):
        iant1 = ant1[i]
        iant2 = ant2[i]
        xyz2 = (ant_position[iant1, :] - ant_position[iant2, :]) ** 2
        baselength[i] = np.sqrt(np.sum(xyz2))

    tmin = np.amin(timestamps)
    timestamps1 = timestamps - tmin
    timestamps1 = timestamps1 / 60.0
    tmax = np.amax(timestamps1)

    baselength_log10 = np.log10(baselength)
    blmax_log10 = np.amax(baselength_log10)
    blmin_log10 = np.amin(baselength_log10)
    log.info(f"Log values min, max , {blmin_log10}, {blmax_log10}")

    # Plot dimensions
    nxt = int(np.amax(timestamps1))
    nx = len(np.unique(timestamps1))
    ny = int(nxt)

    visdata_log = np.zeros((nx, ny, 4))
    viscount_log = np.zeros((nx, ny, 4))
    # Loop over visibilities
    for i in range(baselength.shape[0]):
        ix = int(np.rint(timestamps1[i] / tmax * (nx - 1)))
        iyl = int(
            np.rint(
                (baselength_log10[i] - blmin_log10)
                / (blmax_log10 - blmin_log10)
                * (ny - 1)
            )
        )
        visdata_log[ix, iyl, :] = visdata_log[ix, iyl, :] + np.abs(ms[i, nch, :])
        viscount_log[ix, iyl, :] = viscount_log[ix, iyl, :] + 1

    for i in range(nx):
        for j in range(ny):
            if viscount_log[i, j, 0] > 0.0:
                visdata_log[i, j, :] = visdata_log[i, j, :] / viscount_log[i, j, :]

    title_full = msname + "\n" + title_part2
    plt.clf()
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title_full, fontsize=20.0)
    sptitle = ["XX", "XY", "YX", "YY"]

    for col in range(2):
        for row in range(2):
            ax = axs[row, col]
            idx = 2 * col + row
            im = ax.imshow(
                visdata_log[:, :, idx],
                extent=[blmin_log10, blmax_log10, tmax, 0],
                cmap="nipy_spectral",
            )
            ax.set_aspect((blmax_log10 - blmin_log10) / tmax)
            ax.set_title(sptitle[idx], fontsize=14.0)
            if idx == 1 or idx == 3:
                ax.set_xlabel("Base length, log10(m)", fontsize=13.0)
            if idx == 0 or idx == 1:
                ax.set_ylabel("Time, min", fontsize=13.0)
            cbar = fig.colorbar(im, ax=ax)
            if idx == 2 or idx == 3:
                cbar.ax.set_ylabel("Flux, Jy", rotation=270, fontsize=13.0, labelpad=13)

    figure_name = str(os.path.splitext(msname)[0]) + "_vis.png"
    plt.savefig(figure_name)
    plt.clf()
    log.info(f"Exporting plot to {figure_name}")

    return True
