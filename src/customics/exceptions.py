"""Custom exceptions for the customics package."""


class CustOmicsError(Exception):
    """Base exception for all customics errors."""


class DataValidationError(CustOmicsError):
    """Raised when input data fails validation checks."""


class ModelNotFittedError(CustOmicsError):
    """Raised when inference is attempted before calling fit()."""


class ConfigurationError(CustOmicsError):
    """Raised when model or training configuration is invalid."""
