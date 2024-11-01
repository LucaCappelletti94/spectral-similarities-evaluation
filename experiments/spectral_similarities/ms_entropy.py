"""Implementation of the Spectral Similarity interface for the Mass Spec Entropy method."""

import numpy as np
from ms_entropy import (
    calculate_unweighted_entropy_similarity,
    calculate_entropy_similarity,
)
from matchms import Spectrum

from experiments.spectral_similarities.spectral_similarity import SpectralSimilarity


class UnweightedMassSpecEntropy(SpectralSimilarity):
    """Implementation of the Spectral Similarity interface for the Mass Spec Entropy method."""

    def __init__(self, tolerance: float, verbose: bool, n_jobs: int = 1):
        """Initialize the ModifiedCosine similarity measure."""
        super().__init__(verbose, n_jobs)
        self.tolerance = tolerance

    def name(self) -> str:
        """Return name of the ModifiedCosine similarity measure."""
        return "Unweighted MS Entropy"

    def compute_similarity(self, spectrum1: Spectrum, spectrum2: Spectrum) -> float:
        """Compute similarity between two spectra."""

        # We convert the spectra to two NumPy arrays with shape
        # (n, 2), where n is the number of peaks in the spectrum,
        # and the second dimension contains the m/z and intensity.
        spectrum1_array: np.ndarray = np.stack(
            [spectrum1.peaks.mz, spectrum1.peaks.intensities]
        )

        spectrum2_array: np.ndarray = np.stack(
            [spectrum2.peaks.mz, spectrum2.peaks.intensities]
        )

        return calculate_unweighted_entropy_similarity(
            spectrum1_array,
            spectrum2_array,
            ms2_tolerance_in_ppm=self.tolerance,
            clean_spectra=True,
        )

    def to_dict(self) -> dict:
        """Return the ModifiedCosine similarity measure as a dictionary."""
        return {
            "name": self.name(),
            "tolerance": self.tolerance,
        }


class WeightedMassSpecEntropy(SpectralSimilarity):
    """Implementation of the Spectral Similarity interface for the Mass Spec Entropy method."""

    def __init__(self, tolerance: float, verbose: bool, n_jobs: int = 1):
        """Initialize the ModifiedCosine similarity measure."""
        super().__init__(verbose, n_jobs)
        self.tolerance = tolerance

    def name(self) -> str:
        """Return name of the ModifiedCosine similarity measure."""
        return "Weighted MS Entropy"

    def compute_similarity(self, spectrum1: Spectrum, spectrum2: Spectrum) -> float:
        """Compute similarity between two spectra."""

        # We convert the spectra to two NumPy arrays with shape
        # (n, 2), where n is the number of peaks in the spectrum,
        # and the second dimension contains the m/z and intensity.
        spectrum1_array: np.ndarray = np.stack(
            [spectrum1.peaks.mz, spectrum1.peaks.intensities]
        )

        spectrum2_array: np.ndarray = np.stack(
            [spectrum2.peaks.mz, spectrum2.peaks.intensities]
        )

        return calculate_entropy_similarity(
            spectrum1_array,
            spectrum2_array,
            ms2_tolerance_in_ppm=self.tolerance,
            clean_spectra=True,
        )

    def to_dict(self) -> dict:
        """Return the ModifiedCosine similarity measure as a dictionary."""
        return {
            "name": self.name(),
            "tolerance": self.tolerance,
        }
