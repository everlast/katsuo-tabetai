"""Kochi katsuo restaurant ranking agent."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("katsuo-tabetai")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.1.0"
