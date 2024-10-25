"""Executor for the experiment."""

from argparse import ArgumentParser
from multiprocessing import cpu_count
import pandas as pd
from experiments import experiment


def main():
    """Run the experiment."""
    parser = ArgumentParser(description="Run the experiment.")
    parser.add_argument(
        "--quantity",
        type=int,
        required=True,
        help="The number of spectra to sample.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        required=True,
        help="The random state to use.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        required=True,
        help="The number of iterations to run.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="The output file to save the results to.",
    )
    parser.add_argument(
        "--data-directory",
        type=str,
        required=True,
        help="The directory to store the datasets in.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print additional information.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=cpu_count(),
        help="The number of jobs to use.",
    )
    args = parser.parse_args()

    results: pd.DataFrame = experiment(
        iterations=args.iterations,
        quantity=args.quantity,
        random_state=args.random_state,
        directory=args.data_directory,
        verbose=args.verbose,
        n_jobs=args.n_jobs,
        cache=True,
    )

    results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
