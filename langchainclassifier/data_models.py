import json
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class TransactionCategory(BaseModel):
    beam_label: Optional[str] = Field(
        description="The primary category of the transaction"
    )
    beam_tag: Optional[str] = Field(description="The specific tag for the transaction")


class TransactionClassification(BaseModel):
    categories: List[TransactionCategory] = Field(
        description="List of transaction categories"
    )
    reasoning: str = Field(description="Explanation for the classification")

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, v):
        if not v:
            raise ValueError("At least one category must be provided")
        if len(v) > 1:
            raise ValueError("Only one category should be provided per transaction")
        return v


class ClassificationResult(BaseModel):
    index: int = Field(description="Index of the transaction in the dataset")
    transaction_data: dict = Field(description="Original transaction data")
    classification: Optional[TransactionClassification] = Field(
        description="Classification results"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if classification failed"
    )


class BatchClassificationResults(BaseModel):
    results: List[ClassificationResult] = Field(
        description="List of classification results"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp of the classification batch",
    )
    success_rate: float = Field(description="Percentage of successful classifications")

    def to_json(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump(self.model_dump(), f, indent=2)
