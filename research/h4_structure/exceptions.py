"""H4 structure research exceptions."""


class H4StructureError(Exception):
    """Base error for H4 structure research."""


class H4StructureInputError(H4StructureError):
    """Missing or invalid inputs."""
