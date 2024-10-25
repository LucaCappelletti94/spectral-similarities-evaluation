"""Submodule providing interface and implementation of spectral similarities."""

from experiments.spectral_similarities.spectral_similarity import SpectralSimilarity
from experiments.spectral_similarities.matchms_similarities import (
    CosineGreedy,
    NeutralLossesCosine,
    ModifiedCosine,
)

__all__ = [
    "SpectralSimilarity",
    "CosineGreedy",
    "NeutralLossesCosine",
    "ModifiedCosine",
]
