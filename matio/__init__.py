"""Python library for performing I/O (currently read-only) operations on MATLAB MAT-files."""

from .mat_opaque_tools import MatioOpaque
from .matio_base import load_from_mat, save_to_mat
from .subsystem import get_matio_context

__all__ = ["load_from_mat", "save_to_mat", "get_matio_context", "MatioOpaque"]
