"""Implementation of the SpectralDataset class for the synthetic dataset."""

import os
import pickle
from downloaders import BaseDownloader
from matchms import Spectrum
from matchms.filtering import normalize_intensities, default_filters
from experiments.datasets.spectral_dataset import Dataset


class SyntheticDataset(Dataset):
    """Implementation of the SpectralDataset class for the synthetic dataset."""

    def _load_spectra(self) -> list[Spectrum]:
        """Load the synthetic dataset."""
        downloader = BaseDownloader(
            verbose=self.verbose,
            target_directory=self.directory,
        )

        # Download the synthetic dataset
        downloader.download(
            "https://zenodo.org/records/8287341/files/isdb_pos_cleaned.pkl",
            os.path.join(self.directory, "isdb_pos_cleaned.pkl"),
        )

        with open(os.path.join(self.directory, "isdb_pos_cleaned.pkl"), "rb") as file:
            data: list[Spectrum] = pickle.load(file)

        # We filter and normalize the spectra
        data = [normalize_intensities(default_filters(s)) for s in data]

        return data

    def name(self) -> str:
        """Return the name of the synthetic dataset."""
        return "Synthetic"

    def tolerance(self) -> float:
        """Return the tolerance of the synthetic dataset."""
        return 0.01

    def to_dict(self) -> dict:
        """Return the synthetic dataset as a dictionary."""
        return {
            "name": self.name(),
            "tolerance": self.tolerance(),
        }