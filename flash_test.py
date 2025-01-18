# %%
import json
import os
from datetime import datetime
from typing import List, Optional

import polars as pl
from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm

# Load environment variables
load_dotenv()


# Define your Pydantic models
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


# Define your Pydantic models
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


# %%
class TransactionClassifier:
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        """Initialize the classifier with Google API key and model settings."""
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model_name,
            temperature=0,
            convert_system_message_to_human=True,
        )

        self.parser = PydanticOutputParser(pydantic_object=TransactionClassification)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a financial transaction classifier. Analyze the transaction "
                    "and classify it into appropriate categories. Provide clear reasoning "
                    "for your classification.",
                ),
                (
                    "human",
                    "Transaction details: {transaction}\n\n"
                    "Classify this transaction using the following format:\n{format_instructions}",
                ),
            ]
        )

    def classify_transaction(
        self, transaction_data: dict, index: int
    ) -> ClassificationResult:
        """Classify a single transaction."""
        try:
            # Format the transaction data for the prompt
            formatted_transaction = "\n".join(
                [f"{k}: {v}" for k, v in transaction_data.items()]
            )

            # Create the full prompt with format instructions
            chain = self.prompt | self.llm | self.parser

            # Get the classification
            classification = chain.invoke(
                {
                    "transaction": formatted_transaction,
                    "format_instructions": self.parser.get_format_instructions(),
                }
            )

            return ClassificationResult(
                index=index,
                transaction_data=transaction_data,
                classification=classification,
            )

        except Exception as e:
            return ClassificationResult(
                index=index, transaction_data=transaction_data, error=str(e)
            )

    def classify_batch(
        self, df: pl.DataFrame, batch_size: int = 10
    ) -> BatchClassificationResults:
        """Classify a batch of transactions from a Polars DataFrame."""
        results = []
        total_rows = len(df)

        # Convert DataFrame to list of dictionaries for processing
        transactions = df.to_dicts()

        # Process in batches with progress bar
        for i in tqdm(range(0, total_rows, batch_size), desc="Processing transactions"):
            batch = transactions[i : min(i + batch_size, total_rows)]

            for idx, transaction in enumerate(batch):
                result = self.classify_transaction(transaction, i + idx)
                results.append(result)

        # Calculate success rate
        successful = sum(1 for r in results if r.classification is not None)
        success_rate = (successful / total_rows) * 100

        return BatchClassificationResults(results=results, success_rate=success_rate)


# Example usage
def main():
    # Set up your Google API key
    api_key = os.getenv("GOOGLE_API_KEY")

    # Create sample DataFrame
    df = pl.read_csv(os.getenv("TRANSACTIONS_DATA_PATH")).sample(10)

    # Initialize classifier
    classifier = TransactionClassifier(api_key=api_key)

    # Run classification
    results = classifier.classify_batch(df)

    # Save results
    results.to_json("classification_results.json")

    # Print summary
    print(f"Classification complete. Success rate: {results.success_rate:.2f}%")


if __name__ == "__main__":
    main()

# %%
