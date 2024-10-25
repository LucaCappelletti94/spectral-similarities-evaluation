"""Submodule providing exceptions used in the experiments."""


class ExperimentError(Exception):
    """Base class for exceptions raised by the experiments."""


class UnknownPolarity(ExperimentError):
    """Exception raised when an Unknown polarity is provided."""

    def __init__(self, polarity: str):
        """Initialize the UnknownPolarityError."""
        super().__init__(
            f"Unknown polarity: {polarity}: we only support 'positive', 'negative' and 'both'."
        )


class UnknownApparatus(ExperimentError):
    """Exception raised when an Unknown apparatus is provided."""

    def __init__(self, apparatus: str):
        """Initialize the UnknownApparatusError."""
        super().__init__(
            f"Unknown apparatus: {apparatus}: we only support 'orbitrap', 'qtof' and 'all'."
        )
