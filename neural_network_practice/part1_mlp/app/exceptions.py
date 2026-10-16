class MLPError(Exception):
    """Base exception for expected application errors."""


class InvalidLayerSizeError(MLPError):
    """Raised when layer sizes are invalid."""


class MismatchedDataError(MLPError):
    """Raised when data dimensions are incompatible."""


class DataFileError(MLPError):
    """Raised when a data file cannot be read or parsed."""


class InvalidInputError(MLPError):
    """Raised when a user-provided value is invalid."""


class ModelFileError(MLPError):
    """Raised when model weights cannot be saved or loaded."""