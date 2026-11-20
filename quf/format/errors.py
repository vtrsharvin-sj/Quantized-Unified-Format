"""Error types for reading and validating QUF files."""


class QUFFormatError(Exception):
    """Base class for all QUF file format errors."""


class InvalidMagicError(QUFFormatError):
    def __init__(self, found: bytes, expected: bytes = b"QUF1"):
        self.found = found
        self.expected = expected
        super().__init__(f"invalid magic: expected {expected!r}, found {found!r}")


class UnsupportedVersionError(QUFFormatError):
    def __init__(self, file_version: int, supported_version: int):
        self.file_version = file_version
        self.supported_version = supported_version
        super().__init__(
            f"unsupported version: file version is {file_version}, "
            f"highest supported version is {supported_version}"
        )


class TruncatedHeaderError(QUFFormatError):
    def __init__(self, message: str = "truncated header"):
        super().__init__(message)


class TensorBoundsError(QUFFormatError):
    """A tensor offset is misaligned or its data extends past the end of the file."""

    def __init__(self, tensor_name: str, message: str):
        self.tensor_name = tensor_name
        super().__init__(f"tensor '{tensor_name}': {message}")


class HeaderSchemaError(QUFFormatError):
    """The header is not valid JSON or lacks a required field."""
