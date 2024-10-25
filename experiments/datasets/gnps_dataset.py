"""Submodule implementing the Dataset interface for the GNPS dataset."""

from typing import Optional
import os
from downloaders import BaseDownloader
from matchms import Spectrum
from matchms.importing import load_from_mgf
from matchms.filtering import normalize_intensities, default_filters
from tqdm.auto import tqdm
import pandas as pd
from experiments.datasets.spectral_dataset import Dataset
from experiments.exceptions import UnknownPolarity, UnknownApparatus


class GNPSDataset(Dataset):
    """Implementation of the Dataset interface for the GNPS dataset."""

    def __init__(
        self,
        only_lotus: bool,
        polarity: str,
        apparatus: str,
        directory: str,
        verbose: bool,
    ):
        """Initialize the GNPS dataset.

        Parameters
        ----------
        only_lotus : bool
            Whether to only include spectra that are in the Lotus dataset.
        polarity : str
            The polarity of the spectra to include.
            Can be either "positive", "negative", or "both".
        directory : str
            The directory to store the dataset in.
        verbose : bool
            Whether to print additional information.
        """
        super().__init__(directory, verbose)

        if polarity not in ("positive", "negative", "both"):
            raise UnknownPolarity(polarity)

        if apparatus not in ("qtof", "orbitrap", "all"):
            raise UnknownApparatus(apparatus)

        self._only_lotus: bool = only_lotus
        self._polarity: str = polarity
        self._apparatus: str = apparatus
        self._all_spectra: list[Spectrum] = []

    def _load_spectra(self) -> None:
        """Load the GNPS dataset."""
        downloader = BaseDownloader(
            verbose=self.verbose,
            target_directory=self.directory,
            process_number=1,
        )
        downloader.download(
            [
                "https://external.gnps2.org/processed_gnps_data/matchms.mgf",
                "https://zenodo.org/record/7534071/files/230106_frozen_metadata.csv.gz",
            ],
            [
                os.path.join(self.directory, "matchms.mgf"),
                os.path.join(self.directory, "lotus_metadata.csv.gz"),
            ]
        )

        spectra: list[Spectrum] = []

        if self._only_lotus:
            lotus_inchikeys: set[str] = set(
                pd.read_csv(
                    os.path.join(self.directory, "lotus_metadata.csv.gz"),
                    low_memory=False,
                ).structure_inchikey.values
            )
        else:
            lotus_inchikeys = set()

        for spectrum in tqdm(
            load_from_mgf(os.path.join(self.directory, "matchms.mgf")),
            desc="Loading spectra",
            unit="spectrum",
            dynamic_ncols=True,
            leave=False,
            disable=not self.verbose,
        ):
            smiles: Optional[str] = spectrum.get("smiles")
            if smiles is None:
                continue

            inchikey: Optional[str] = spectrum.get("inchikey")
            if inchikey is None:
                continue

            if self._only_lotus and inchikey not in lotus_inchikeys:
                continue

            if self._polarity != "both" and self._polarity != spectrum.get("ionmode"):
                continue

            if self._apparatus != "all" and self._apparatus != spectrum.get(
                "ms_mass_analyzer"
            ):
                continue

            spectrum: Spectrum = default_filters(spectrum)
            spectrum: Spectrum = normalize_intensities(spectrum)

            spectra.append(spectrum)

        return spectra

    def name(self) -> str:
        """Return the name of the GNPS dataset."""
        descriptors: list[str] = []

        if self._only_lotus:
            descriptors.append("LOTUS")

        if self._polarity == "positive":
            descriptors.append("Positives")

        if self._polarity == "negative":
            descriptors.append("Negatives")

        if self._apparatus == "qtof":
            descriptors.append("QTOF")

        if self._apparatus == "orbitrap":
            descriptors.append("Orbitrap")

        description = ""
        if len(descriptors) > 0:
            description = f" ({', '.join(descriptors)})"

        return f"GNPS{description}"

    def tolerance(self) -> float:
        """Return the tolerance of the GNPS dataset."""
        return 0.1

    def to_dict(self) -> dict:
        """Return the GNPS dataset as a dictionary."""
        return {
            "name": self.name(),
            "tolerance": self.tolerance(),
            "only_lotus": self._only_lotus,
            "polarity": self._polarity,
            "apparatus": self._apparatus,
        }
