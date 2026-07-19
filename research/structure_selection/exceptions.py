"""Structure selection research exceptions."""


class StructureSelectionError(Exception):
    """Base error."""


class StructureSelectionInputError(StructureSelectionError):
    """Missing inputs."""
