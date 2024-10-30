"""Similarity score based on ms2deepscore."""

import os
from matchms import Spectrum
from ms2deepscore import MS2DeepScore as MS2DeepScoreModel
from ms2deepscore.models import load_model
from downloaders import BaseDownloader

from experiments.spectral_similarities.spectral_similarity import SpectralSimilarity


class MS2DeepScore(SpectralSimilarity):
    """Implementation of MS2DeepScore similarity measure."""

    def __init__(self, directory: str, verbose: bool, n_jobs: int = 1) -> None:
        """Initialize MS2DeepScore similarity measure."""
        super().__init__(verbose, n_jobs)

        downloader = BaseDownloader(
            process_number=1,
            verbose=verbose,
        )
        downloader.download(
            "https://zenodo.org/records/13897744/files/ms2deepscore_model.pt?download=1",
            os.path.join(directory, "ms2deepscore_model.pt"),
        )
        self._model: MS2DeepScoreModel = load_model(
            os.path.join(directory, "ms2deepscore_model.pt")
        )

    def name(self) -> str:
        """Return the name of the similarity measure."""
        return "MS2DeepScore"

    def compute_similarity(self, spectrum1: Spectrum, spectrum2: Spectrum) -> float:
        """Compute similarity between two spectra."""
        return self._model.pair(spectrum1, spectrum2)

    def to_dict(self) -> dict:
        """Return the ModifiedCosine similarity measure as a dictionary."""
        return {
            "name": self.name(),
        }
