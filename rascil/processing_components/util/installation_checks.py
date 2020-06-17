"""Function to check the installation

"""

import logging

from rascil.data_models import rascil_data_path

log = logging.getLogger(__file__)

__all__ = ['check_data_directory']

def check_data_directory(verbose=False, fatal=True):
    """ Check the RASCIL data directory to see if it has been installed correctly
    """
    dp = rascil_data_path("")

    try:

        canary = rascil_data_path("configurations/LOWBD2.csv")
        with open(canary, "r") as f:
            first = f.read(1)
            if first == "version https://git-lfs.github.com/spec/v1":
                log.warning("The RASCIL data directory is not filled correctly - git lfs pull is required")
            else:
                if verbose: print("The RASCIL data directory appears to have been installed correctly")
    except FileNotFoundError:
        if fatal:
            log.error("The RASCIL data directory is not available - stopping")
        else:
            log.warning("The RASCIL data directory is not available - continuing but any simulations will fail")


if __name__ == "__main__":
    check_data_directory()
