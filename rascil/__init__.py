
from . import data_models
from . import processing_components

from .processing_components.util.installation_checks import check_data_directory

check_data_directory(fatal=False)
