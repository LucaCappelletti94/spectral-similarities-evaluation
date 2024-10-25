"""Main loop of the experiment."""

from typing import Type
from cache_decorator import Cache
import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange
from matchms import Spectrum
from scipy.stats import pearsonr, spearmanr, kendalltau
from experiments.datasets import Dataset, GNPSDataset, SyntheticDataset
from experiments.spectral_similarities import (
    SpectralSimilarity,
    CosineGreedy,
    NeutralLossesCosine,
    ModifiedCosine,
)
from experiments.molecular_similarities import all_fingerprints, jaccard


@Cache(
    cache_path="results/{_hash}.csv",
    use_approximated_hash=True,
    args_to_ignore=["cache", "verbose", "n_jobs"],
    enable_cache_arg_name="cache",
    capture_enable_cache_arg_name=False,
)
def experiment_step(
    dataset: Type[Dataset],
    similarity_measure: Type[SpectralSimilarity],
    quantity: int,
    random_state: int,
    verbose: bool,
    n_jobs: int,
    cache: bool,  # pylint: disable=unused-argument
) -> pd.DataFrame:
    """Executes a single step of the experiment."""
    rows: list[Spectrum] = dataset.sample_spectra(quantity, random_state)
    columns: list[Spectrum] = dataset.sample_spectra(quantity, random_state)

    rows_smiles: list[str] = [spectrum.get("smiles") for spectrum in rows]
    columns_smiles: list[str] = [spectrum.get("smiles") for spectrum in columns]

    rows_fingerprints: dict[str, np.ndarray] = all_fingerprints(
        rows_smiles, verbose=verbose, n_jobs=n_jobs
    )
    columns_fingerprints: dict[str, np.ndarray] = all_fingerprints(
        columns_smiles, verbose=verbose, n_jobs=n_jobs
    )

    spectral_similarities: np.ndarray = similarity_measure.transform(rows, columns)

    results: list[dict] = []

    for fingerprint_name, rows_fingerprint in tqdm(
        rows_fingerprints.items(),
        desc="Fingerprints",
        unit="fingerprint",
        dynamic_ncols=True,
        leave=False,
        total=len(rows_fingerprints),
        disable=not verbose,
    ):
        fingerprint_similarity = jaccard(
            rows_fingerprint,
            columns_fingerprints[fingerprint_name],
        )
        for correlation_method_name, correlation_method in tqdm(
            (
                ("Pearson", pearsonr),
                ("Spearman", spearmanr),
                ("Kendall", kendalltau),
            ),
            desc="Correlation",
            unit="correlation method",
            dynamic_ncols=True,
            leave=False,
            disable=not verbose,
        ):
            correlation, p_value = correlation_method(
                fingerprint_similarity.flatten(), spectral_similarities.flatten()
            )
            results.append(
                {
                    "dataset": dataset.name(),
                    "fingerprint": fingerprint_name,
                    "spectral_similarity": similarity_measure.name(),
                    "correlation_method": correlation_method_name,
                    "correlation": correlation,
                    "p_value": p_value,
                }
            )

    return pd.DataFrame(results)


def experiment(
    iterations: int,
    quantity: int,
    random_state: int,
    directory: str,
    n_jobs: int,
    verbose: bool,
    cache: bool,
) -> pd.DataFrame:
    """Executes the experiment."""
    datasets: list[Type[Dataset]] = [
        SyntheticDataset(directory=directory, verbose=verbose),
    ]

    for polarity in ["positive", "negative", "both"]:
        for apparatus in ["qtof", "orbitrap", "all"]:
            for only_lotus in [True, False]:
                datasets.append(
                    GNPSDataset(
                        only_lotus=only_lotus,
                        directory=directory,
                        polarity=polarity,
                        apparatus=apparatus,
                        verbose=verbose,
                    )
                )

    results: list[pd.DataFrame] = []

    for dataset in tqdm(
        datasets,
        desc="Datasets",
        unit="dataset",
        dynamic_ncols=True,
        leave=False,
        disable=not verbose,
    ):
        similarity_measures: list[Type[SpectralSimilarity]] = [
            CosineGreedy(tolerance=dataset.tolerance(), verbose=verbose, n_jobs=n_jobs),
            NeutralLossesCosine(
                tolerance=dataset.tolerance(), verbose=verbose, n_jobs=n_jobs
            ),
            ModifiedCosine(
                tolerance=dataset.tolerance(), verbose=verbose, n_jobs=n_jobs
            ),
        ]
        for similarity_measure in tqdm(
            similarity_measures,
            desc="Similarities",
            unit="similarity measure",
            dynamic_ncols=True,
            leave=False,
            disable=not verbose,
        ):
            for iteration in trange(
                iterations,
                desc="Iterations",
                unit="iteration",
                dynamic_ncols=True,
                leave=False,
                disable=not verbose,
            ):
                results.append(
                    experiment_step(
                        dataset=dataset,
                        similarity_measure=similarity_measure,
                        quantity=quantity,
                        random_state=(random_state * (iteration + 1)) % 2**32,
                        verbose=verbose,
                        n_jobs=n_jobs,
                        cache=cache,
                    )
                )

    return pd.concat(results)
