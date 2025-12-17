"""GrOpt Python interface."""

__all__ = []

import importlib.metadata

__version__ = importlib.metadata.version("gropt")

from ._utils import *  # noqa
from ._gropt import gropt  # noqa

from . import _utils

__all__.append("gropt")
__all__.extend(_utils.__all__)  # noqa
