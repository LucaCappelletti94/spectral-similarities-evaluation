"""Implementation of the SpectralSimilarity Interface for the MatchMS similarities."""

from matchms import Spectrum
from matchms.similarity import (
    CosineGreedy as MatchMSCosineGreedy,
    NeutralLossesCosine as MatchMSNeutralLossesCosine,
    ModifiedCosine as MatchMSModifiedCosine,
)
from experiments.spectral_similarities.spectral_similarity import SpectralSimilarity


class CosineGreedy(SpectralSimilarity):
    """Implementation of the CosineGreedy similarity measure."""

    def __init__(self, tolerance: float, verbose: bool, n_jobs: int = 1):
        """Initialize the CosineGreedy similarity measure."""
        super().__init__(verbose, n_jobs)
        self._cosine_greedy = MatchMSCosineGreedy(tolerance=tolerance)

    def name(self) -> str:
        """Return name of the CosineGreedy similarity measure."""
        return "Greedy Cosine"

    def compute_similarity(self, spectrum1: Spectrum, spectrum2: Spectrum) -> float:
        """Compute similarity between two spectra."""
        return self._cosine_greedy.pair(spectrum1, spectrum2)[()][0]

    def to_dict(self) -> dict:
        """Return the CosineGreedy similarity measure as a dictionary."""
        return {
            "name": self.name(),
            "tolerance": self._cosine_greedy.tolerance,
        }


class NeutralLossesCosine(SpectralSimilarity):
    """Implementation of the NeutralLossesCosine similarity measure."""

    def __init__(self, tolerance: float, verbose: bool, n_jobs: int = 1):
        """Initialize the NeutralLossesCosine similarity measure."""
        super().__init__(verbose, n_jobs)
        self._neutral_losses_cosine = MatchMSNeutralLossesCosine(
            tolerance=tolerance,
        )

    def name(self) -> str:
        """Return name of the NeutralLossesCosine similarity measure."""
        return "Neutral Losses Cosine"

    def compute_similarity(self, spectrum1: Spectrum, spectrum2: Spectrum) -> float:
        """Compute similarity between two spectra."""
        return self._neutral_losses_cosine.pair(spectrum1, spectrum2)[()][0]

    def to_dict(self) -> dict:
        """Return the NeutralLossesCosine similarity measure as a dictionary."""
        return {
            "name": self.name(),
            "tolerance": self._neutral_losses_cosine.tolerance,
        }


class ModifiedCosine(SpectralSimilarity):
    """Implementation of the ModifiedCosine similarity measure."""

    def __init__(self, tolerance: float, verbose: bool, n_jobs: int = 1):
        """Initialize the ModifiedCosine similarity measure."""
        super().__init__(verbose, n_jobs)
        self._modified_cosine = MatchMSModifiedCosine(tolerance=tolerance)

    def name(self) -> str:
        """Return name of the ModifiedCosine similarity measure."""
        return "Modified Cosine"

    def compute_similarity(self, spectrum1: Spectrum, spectrum2: Spectrum) -> float:
        """Compute similarity between two spectra."""
        return self._modified_cosine.pair(spectrum1, spectrum2)[()][0]

    def to_dict(self) -> dict:
        """Return the ModifiedCosine similarity measure as a dictionary."""
        return {
            "name": self.name(),
            "tolerance": self._modified_cosine.tolerance,
        }
