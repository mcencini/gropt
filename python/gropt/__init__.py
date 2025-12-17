"""GrOpt Python interface."""

__all__ = []

import importlib.metadata

__version__ = importlib.metadata.version("gropt")

from ._gropt import gropt  # noqa

__all__.append("gropt")
