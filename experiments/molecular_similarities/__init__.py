"""Submodule defining utilities for molecular similarities."""

from typing import List, Tuple, Type
from numba import njit, prange
import numpy as np
from tqdm.auto import tqdm
from skfp.bases import BaseFingerprintTransformer
from skfp.fingerprints.ecfp import ECFPFingerprint
from skfp.fingerprints.avalon import AvalonFingerprint
from skfp.fingerprints.layered import LayeredFingerprint
from skfp.fingerprints.rdkit_fp import RDKitFingerprint


@njit(parallel=True)
def jaccard(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Calculate the similarities between the rows and columns of the fingerprints."""
    similarity = np.zeros(
        (
            rows.shape[0],
            columns.shape[0],
        ),
        dtype=np.float32,
    )
    for i in prange(rows.shape[0]):  # pylint: disable=not-an-iterable
        row = rows[i]
        for j in range(columns.shape[0]):
            column = columns[j]
            for k in range(row.shape[0]):
                if row[k] == column[k]:
                    similarity[i, j] += 1
            similarity[i, j] /= row.shape[0]
    return similarity


def all_fingerprints(
    smiles: list[str], verbose: bool, n_jobs: int
) -> dict[str, np.ndarray]:
    """Computes all predefined fingerprints for the given SMILES."""
    fingerprints: list[Type[BaseFingerprintTransformer]] = [
        ECFPFingerprint(fp_size=2048, verbose=False, n_jobs=n_jobs),
        AvalonFingerprint(fp_size=2048, verbose=False, n_jobs=n_jobs),
        LayeredFingerprint(fp_size=2048, verbose=False, n_jobs=n_jobs),
        RDKitFingerprint(fp_size=2048, verbose=False, n_jobs=n_jobs),
    ]

    fingerprint_matrices: dict[str, np.ndarray] = {}

    for fingerprint in tqdm(
        fingerprints,
        desc="Fingerprints",
        unit="fingerprint",
        dynamic_ncols=True,
        leave=False,
        total=len(fingerprints),
        disable=not verbose,
    ):
        fingerprint_matrices[fingerprint.__class__.__name__] = (
            fingerprint.fit_transform(smiles)
        )

    return fingerprint_matrices
