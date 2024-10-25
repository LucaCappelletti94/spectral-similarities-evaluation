"""Test run of the experimental pipeline."""

from multiprocessing import cpu_count
from experiments import experiment


def test_run():
    """Test the experimental pipeline."""
    experiment(
        iterations=1,
        quantity=5,
        random_state=42,
        directory="data",
        n_jobs=cpu_count(),
        verbose=True,
        cache=False,
    )
