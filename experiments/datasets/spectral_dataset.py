"""Submodule defining the interface for a spectral dataset."""

from abc import abstractmethod
from matchms import Spectrum
import numpy as np
from dict_hash import Hashable, sha256


class Dataset(Hashable):
    """Interface for spectral datasets."""

    def __init__(self, directory: str, verbose: bool):
        """Initialize the spectral dataset."""
        self._directory: str = directory
        self._verbose: bool = verbose
        self._spectra: list[Spectrum] = []

    @property
    def verbose(self) -> bool:
        """Return verbose mode."""
        return self._verbose

    @property
    def directory(self) -> str:
        """Return the directory of the dataset."""
        return self._directory

    @abstractmethod
    def name(self) -> str:
        """Return name of the dataset."""

    @abstractmethod
    def _load_spectra(self) -> list[Spectrum]:
        """Return the spectra in the dataset."""

    def spectra(self) -> list[Spectrum]:
        """Return the spectra in the dataset."""
        if not self._spectra:
            self._spectra = self._load_spectra()
        return self._spectra

    @abstractmethod
    def tolerance(self) -> float:
        """Return the tolerance of the dataset."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Return the dataset as a dictionary."""

    def consistent_hash(self, use_approximation: bool = False) -> str:
        """Return a consistent hash of the dataset."""
        return sha256(self.to_dict(), use_approximation=use_approximation)

    def sample_spectra(self, quantity: int, random_state: int) -> list[Spectrum]:
        """Return a random sample of the spectra."""
        rng = np.random.default_rng(random_state)
        spectra = self.spectra()
        return rng.choice(spectra, size=quantity, replace=False).tolist()
