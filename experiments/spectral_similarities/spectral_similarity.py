"""Submodule providing an interface defining spectral similarities."""

from abc import abstractmethod
from multiprocessing import Pool
from matchms import Spectrum
from tqdm.auto import tqdm
import numpy as np
from dict_hash import Hashable, sha256


class SpectralSimilarity(Hashable):
    """Interface for spectral similarity measures."""

    def __init__(self, verbose: bool, n_jobs: int = 1):
        """Initialize the spectral similarity measure."""
        self._verbose = verbose
        self._n_jobs = n_jobs

    @abstractmethod
    def name(self) -> str:
        """Return name of the spectral similarity measure."""

    @property
    def verbose(self) -> bool:
        """Return verbose mode."""
        return self._verbose

    @property
    def n_jobs(self) -> int:
        """Return number of jobs."""
        return self._n_jobs

    @abstractmethod
    def compute_similarity(self, spectrum1: Spectrum, spectrum2: Spectrum) -> float:
        """Compute similarity between two spectra."""

    def _compute_similarities(self, args) -> np.ndarray:
        """Compute similarity between two spectra."""
        rows, columns = args
        spectra_similarity = np.zeros((len(rows), len(columns)), dtype=np.float32)
        for i, row_spectrum in enumerate(rows):
            for j, column_spectrum in enumerate(columns):
                spectra_similarity[i, j] = self.compute_similarity(row_spectrum, column_spectrum)
        return spectra_similarity

    def transform(self, rows: list[Spectrum], columns: list[Spectrum]) -> np.ndarray:
        """Calculate the similarities between the rows and columns of the spectra."""
        spectra_similarity: np.ndarray = np.zeros(
            (
                len(rows),
                len(columns),
            ),
            dtype=np.float32,
        )

        with Pool(self.n_jobs) as pool:
            chunk_size = len(rows) // self.n_jobs
            tasks = (
                (
                    rows[chunk_number * chunk_size : (chunk_number + 1) * chunk_size],
                    columns,
                )
                for chunk_number in range(self.n_jobs)
            )
            for i, similarities_chunk in enumerate(
                tqdm(
                    pool.imap(self._compute_similarities, tasks),
                    desc=self.name(),
                    leave=False,
                    dynamic_ncols=True,
                    disable=not self.verbose,
                    unit="spectral chunk",
                    total=self.n_jobs,
                )
            ):
                spectra_similarity[i * chunk_size : (i + 1) * chunk_size] = (
                    similarities_chunk
                )
        return spectra_similarity

    @abstractmethod
    def to_dict(self) -> dict:
        """Return the spectral similarity measure as a dictionary."""

    def consistent_hash(self, use_approximation: bool = False) -> str:
        """Return a consistent hash of the spectral similarity measure."""
        return sha256(self.to_dict(), use_approximation=use_approximation)
