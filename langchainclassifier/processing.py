# %%
import json
import os
from typing import Dict, List

import polars as pl
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.progress import Progress
from tenacity import retry, stop_after_attempt, wait_exponential

from langchainclassifier.data_models import (
    BatchClassificationResults,
    ClassificationResult,
    TransactionClassification,
)


class BatchProcessor:
    def __init__(
        self,
        model_name: str = None,
        batch_size: int = 25,
        max_retries: int = 3,
        df=None,
    ):
        if model_name is None:
            raise ValueError("Model Name must be provided")
        # Check if claude is in the model_name
        if "claude" in model_name:
            self.model_name = model_name
            self.llm = self._create_llm(model_provider="anthropic")
        elif "gemini" in model_name:
            self.model_name = model_name
            self.llm = self._create_llm(model_provider="gemini")

        self.batch_size = batch_size
        self.max_retries = max_retries
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

    def _create_llm(self, model_provider: str):
        if model_provider == "anthropic":
            return ChatAnthropic(
                model=self.model_name,
                temperature=0,
                max_tokens_to_sample=1000,
                api_key=os.getenv("CLAUDE_API_KEY"),
            )
        elif model_provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=0,
                max_tokens_to_sample=1000,
                api_key=os.getenv("GOOGLE_API_KEY"),
            )
        else:
            raise ValueError("Invalid model provider")

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
        categories = """
        beam_label:
            ['Loan'],
            ['Create DCA'],
            ['Exchange'],
            ['Income'],
            ['Staking'],
            [None],
            ['Cost']

        beam_tag:
            [None],
            ['Borrow'],
            ['Airdrop'],
            ['Unstake'],
            ['Burn'],
            ['Stake'],
            ['Mint'],
            ['Fee'],
            ['Wrap/Bridge'],
            ['Trade'],
            ['Collateralize'],
            ['Payment']

        please return a json object with the following format:
        {
            "beam_label": "<label>",
            "beam_tag": "<tag>"
        }
        """

        # Process in batches
        all_results = []
        correct_labels = 0
        correct_tags = 0
        processed_count = 0

        with Progress() as progress:
            # Create progress bars
            task_overall = progress.add_task(
                "[cyan]Processing transactions...", total=total_transactions
            )
            task_label_accuracy = progress.add_task(
                "[green]Beam Label Accuracy", total=100
            )
            task_tag_accuracy = progress.add_task(
                "[yellow]Beam Tag Accuracy", total=100
            )

            for start_idx in range(0, total_transactions, self.batch_size):
                end_idx = min(start_idx + self.batch_size, total_transactions)
                batch_transactions = transactions[start_idx:end_idx]

                # Process each transaction in the batch
                batch_results = []
                for idx, transaction in enumerate(batch_transactions):
                    result = self._process_transaction(
                        transaction, categories, start_idx + idx
                    )
                    batch_results.append(result)
                    all_results.append(result)

                    # Update accuracy metrics
                    if result.classification and result.classification.categories:
                        category = result.classification.categories[0]
                        if category.beam_label == df["Beam Label"][start_idx + idx]:
                            correct_labels += 1
                        if category.beam_tag == df["Beam Tag"][start_idx + idx]:
                            correct_tags += 1

                    processed_count += 1

                    # Update progress bars
                    progress.update(
                        task_overall,
                        completed=processed_count,
                        description=f"[cyan]Processing transactions... ({processed_count}/{total_transactions})",
                    )

                    label_accuracy = (correct_labels / processed_count) * 100
                    tag_accuracy = (correct_tags / processed_count) * 100

                    progress.update(
                        task_label_accuracy,
                        completed=label_accuracy,
                        description=f"[green]Beam Label Accuracy: {label_accuracy:.1f}%",
                    )
                    progress.update(
                        task_tag_accuracy,
                        completed=tag_accuracy,
                        description=f"[yellow]Beam Tag Accuracy: {tag_accuracy:.1f}%",
                    )

        # Calculate final success rate (processing success, not accuracy)
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

    # predicted_beam_label = []
    # predicted_beam_tag = []
    rows = []
    for result in batch_results.results:
        row = {
            "index": result.index,
            "actual_beam_label": original_df["Beam Label"][result.index],
            "actual_beam_tag": original_df["Beam Tag"][result.index],
            "id": original_df["id"][result.index],
            "transaction_detail_id": original_df["transaction_detail_id"][result.index],
            "date": original_df["date"][result.index],
            "in_amount": original_df["in_amount"][result.index],
            "in_from": original_df["in_from"][result.index],
            "in_token": original_df["in_token"][result.index],
            "out_amount": original_df["out_amount"][result.index],
            "out_to": original_df["out_to"][result.index],
            "out_token": original_df["out_token"][result.index],
            "fee_amount": original_df["fee_amount"][result.index],
            "fee_paid_by": original_df["fee_paid_by"][result.index],
            "exchange": original_df["exchange"][result.index],
            "API Label": original_df["API Label"][result.index],
            "transaction_hash": original_df["transaction_hash"][result.index],
            "wallet_id": original_df["wallet_id"][result.index],
            "wallet_address": original_df["wallet_address"][result.index],
            "chain": original_df["chain"][result.index],
            "is_current": original_df["is_current"][result.index],
            "description": original_df["description"][result.index],
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
    print("\n===============================================")
    print("Accuracy Metrics:")
    print(f"Beam Label Accuracy: {label_accuracy:.2f}%")
    print(f"Beam Tag Accuracy: {tag_accuracy:.2f}%")
    print("===============================================\n")

    return comparison_df
