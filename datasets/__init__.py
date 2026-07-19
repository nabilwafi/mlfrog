"""Dataset builder layer — FeatureSet ⊕ LabelSet → train-ready splits."""

from datasets.entities import Dataset
from datasets.exceptions import (
    AlignmentError,
    DatasetError,
    DatasetRepositoryError,
    DatasetValidationError,
)
from datasets.pipelines import DatasetPipeline
from datasets.repositories import DatasetRepository
from datasets.services import DatasetService
from datasets.validators import DatasetValidator

__all__ = [
    "AlignmentError",
    "Dataset",
    "DatasetError",
    "DatasetPipeline",
    "DatasetRepository",
    "DatasetRepositoryError",
    "DatasetService",
    "DatasetValidationError",
    "DatasetValidator",
]
