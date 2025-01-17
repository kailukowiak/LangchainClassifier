# %%
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import polars as pl
from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.notebook import tqdm


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


class BatchProcessor:
    def __init__(
        self,
        model_name: str = "claude-3-sonnet-20240229",
        batch_size: int = 25,
        max_retries: int = 3,
        df=None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.llm = self._create_llm()
        self.parser = PydanticOutputParser(pydantic_object=TransactionClassification)
        self.chain = self._create_chain()
        if df is not None:
            example = (
                df.group_by(["Beam Tag", "Beam Label"])
                .head(2)
                .sort(["Beam Tag", "Beam Label"])
                .drop("JSON_Info")
                .to_dicts()
            )
            self.example = f"""Here are some example transactions to help you get started:

                {json.dumps(example, indent=2)}"""
        else:
            self.example = ""

    def _create_llm(self):
        return ChatAnthropic(
            model=self.model_name,
            temperature=0,
            max_tokens_to_sample=1000,
            api_key=os.getenv("CLAUDE_API_KEY"),
        )

    def _create_chain(self):
        template = """You are an expert crypto accountant. Please analyze the following transaction and classify it into appropriate categories. Focus ONLY on the specific transaction provided.

Transaction:
{transactions}

Available Categories:
{categories}

{example}

Please provide:
1. A detailed explanation of your classification reasoning
2. The final classification in the specified format

IMPORTANT: Provide a single classification for this specific transaction only.

{format_instructions}"""

        prompt = ChatPromptTemplate.from_template(template)
        return prompt | self.llm | self.parser

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry_error_callback=lambda retry_state: None,
    )
    def _process_transaction(
        self, transaction: Dict, categories: List[Dict], index: int
    ) -> ClassificationResult:
        try:
            # Format single transaction
            transaction_str = json.dumps(transaction, indent=2)

            # Run the classification
            result = self.chain.invoke(
                {
                    "transactions": transaction_str,
                    "categories": categories,
                    "example": self.example,
                    "format_instructions": self.parser.get_format_instructions(),
                }
            )

            return ClassificationResult(
                index=index, transaction_data=transaction, classification=result
            )
        except Exception as e:
            return ClassificationResult(
                index=index,
                transaction_data=transaction,
                classification=None,
                error=str(e),
            )

    def process_dataset(
        self,
        df: pl.DataFrame,
        drop_cols: List[str] = ["JSON_Info", "Beam Label", "Beam Tag"],
    ) -> BatchClassificationResults:
        # Prepare data
        df_processed = df.drop(drop_cols)
        transactions = df_processed.to_dicts()
        total_transactions = len(transactions)

        # Define categories
        categories = [
            {"Beam Label": "Loan", "Beam Tag": "Payment"},
            {"Beam Label": "Exchange", "Beam Tag": "Trade"},
            {"Beam Label": "Loan", "Beam Tag": None},
            {"Beam Label": "Cost", "Beam Tag": "Fee"},
            {"Beam Label": "Exchange", "Beam Tag": "Wrap/Bridge"},
            {"Beam Label": "Create DCA", "Beam Tag": None},
            {"Beam Label": "Loan", "Beam Tag": "Collateralize"},
            {"Beam Label": "Loan", "Beam Tag": "Borrow"},
            {"Beam Label": "Staking", "Beam Tag": "Stake"},
            {"Beam Label": "Exchange", "Beam Tag": "Mint"},
            {"Beam Label": "Income", "Beam Tag": "Airdrop"},
            {"Beam Label": "Staking", "Beam Tag": "Unstake"},
            {"Beam Label": None, "Beam Tag": None},
            {"Beam Label": "Cost", "Beam Tag": "Burn"},
        ]

        # Process in batches
        all_results = []

        for start_idx in tqdm(
            range(0, total_transactions, self.batch_size), desc="Processing batches"
        ):
            end_idx = min(start_idx + self.batch_size, total_transactions)
            batch_transactions = transactions[start_idx:end_idx]

            # Process each transaction in the batch
            for idx, transaction in enumerate(batch_transactions):
                result = self._process_transaction(
                    transaction, categories, start_idx + idx
                )
                all_results.append(result)

        # Calculate success rate
        success_rate = (
            len([r for r in all_results if r.classification is not None])
            / total_transactions
        )

        return BatchClassificationResults(
            results=all_results, success_rate=success_rate
        )


def create_comparison_df(
    batch_results: BatchClassificationResults, original_df: pl.DataFrame
) -> pl.DataFrame:
    """Create a comparison dataframe with predicted and actual labels"""

    rows = []
    for result in batch_results.results:
        row = {
            "index": result.index,
            "actual_beam_label": original_df["Beam Label"][result.index],
            "actual_beam_tag": original_df["Beam Tag"][result.index],
            "predicted_beam_label": None,
            "predicted_beam_tag": None,
            "correct_label": False,
            "correct_tag": False,
            "reasoning": None,
            "error": result.error,
        }

        if result.classification and result.classification.categories:
            category = result.classification.categories[0]
            row.update(
                {
                    "predicted_beam_label": category.beam_label,
                    "predicted_beam_tag": category.beam_tag,
                    "reasoning": result.classification.reasoning,
                }
            )

            row["correct_label"] = (
                row["predicted_beam_label"] == row["actual_beam_label"]
            )
            row["correct_tag"] = row["predicted_beam_tag"] == row["actual_beam_tag"]

        rows.append(row)

    comparison_df = pl.DataFrame(rows)

    # Calculate accuracy metrics
    label_accuracy = comparison_df["correct_label"].sum() / len(comparison_df) * 100
    tag_accuracy = comparison_df["correct_tag"].sum() / len(comparison_df) * 100

    print("\nAccuracy Metrics:")
    print(f"Beam Label Accuracy: {label_accuracy:.2f}%")
    print(f"Beam Tag Accuracy: {tag_accuracy:.2f}%")

    return comparison_df


def main():
    # Load environment variables and data
    load_dotenv()
    data_path = os.getenv("TRANSACTIONS_DATA_PATH")
    df = pl.read_csv(data_path)

    df = df.group_by(["Beam Tag", "Beam Label"]).head(50)
    # randomly sort the data
    df = df.shuffle()

    # Initialize processor with desired parameters
    processor = BatchProcessor(batch_size=25, max_retries=3, df=df)

    # Process dataset
    results = processor.process_dataset(df)

    # Save results
    results.to_json("out_data/batch_classification_results.json")

    # Create and save comparison DataFrame
    comparison_df = create_comparison_df(results, df)
    comparison_df.write_csv("out_data/batch_classification_comparison.csv")


# %%

if __name__ == "__main__":
    main()
# %%

# tc = TransactionCategory(beam_label="Loan", beam_tag="Payment")
# # %%
# tc.model_dump()
# %%

# data_path = os.getenv("TRANSACTIONS_DATA_PATH")
# df = pl.read_csv(data_path)
# # %%
# # Test out a single transaction that contains staking
# transaction = (
#     df.filter(pl.col("Beam Label") == "Staking")
#     # .drop("JSON_Info", "Beam Label", "Beam Tag")
#     .head(3)
# )
# transaction

# processor = BatchProcessor(batch_size=1, max_retries=3)
# results = processor.process_dataset(transaction)

# # %%
# comparison_df = create_comparison_df(results, transaction)
# # %%
# comparison_df
# # %%
# comparison_df.write_csv("out_data/staking_comparison.csv")
# # %%
# df_samples = (
#     df.group_by(["Beam Tag", "Beam Label"])
#     .head(2)
#     .sort(["Beam Tag", "Beam Label"])
#     .drop("JSON_Info")
# )
# print(f"Sample size: {len(df_samples)}")
# df_samples
# # %%
