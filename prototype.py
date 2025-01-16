# %%
import json
import os
from datetime import datetime
from typing import List, Optional

import polars as pl
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field


# Define the output schema
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

    @property
    def successful_classifications(self) -> int:
        return len([r for r in self.results if r.classification is not None])

    @property
    def failed_classifications(self) -> int:
        return len([r for r in self.results if r.classification is None])

    def to_json(self, filepath: str) -> None:
        """Save results to a JSON file"""
        with open(filepath, "w") as f:
            json.dump(self.dict(), f, indent=2)


def create_classifier_chain():
    # Initialize the LLM
    llm = ChatAnthropic(
        model="claude-3-sonnet-20240229",
        temperature=0,
        max_tokens_to_sample=1000,
        api_key=os.getenv("CLAUDE_API_KEY"),
    )

    # Create output parser
    parser = PydanticOutputParser(pydantic_object=TransactionClassification)

    # Define the prompt template
    template = """You are an expert crypto accountant. Please analyze the following transaction and classify it into appropriate categories.

Transaction Details:
{transaction}

Available Categories:
{categories}

Please provide:
1. A detailed explanation of your classification reasoning
2. The final classification in the specified format

{format_instructions}"""

    prompt = ChatPromptTemplate.from_template(template)

    # Create the chain
    chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser, verbose=True)

    return chain


def classify_transaction(
    df, idx, chain, parser, drop_cols=["JSON_Info", "Beam Label", "Beam Tag"]
):
    # Get transaction details
    transaction_data = df.drop(drop_cols)[idx, :].to_dicts()

    # Define available categories
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

    try:
        # Run the classification
        result = chain.run(
            transaction=transaction_data,
            categories=categories,
            format_instructions=parser.get_format_instructions(),
        )

        classification_result = ClassificationResult(
            index=idx,
            transaction_data=transaction_data[0],  # Get first dict from list
            classification=result,
        )

    except Exception as e:
        error_msg = str(e)
        print(f"Error classifying transaction {idx}: {error_msg}")
        classification_result = ClassificationResult(
            index=idx,
            transaction_data=transaction_data[0],
            classification=None,
            error=error_msg,
        )

    return classification_result


def process_batch(df, start_idx, end_idx):
    """Process a batch of transactions"""
    chain = create_classifier_chain()
    parser = PydanticOutputParser(pydantic_object=TransactionClassification)
    results = []

    for idx in range(start_idx, min(end_idx, len(df))):
        result = classify_transaction(df, idx, chain, parser)
        results.append(result)

    # Calculate success rate
    total = len(results)
    successful = len([r for r in results if r.classification is not None])
    success_rate = (successful / total) if total > 0 else 0.0

    # Create batch results
    batch_results = BatchClassificationResults(
        results=results, success_rate=success_rate
    )

    return batch_results


# Example usage
if __name__ == "__main__":
    load_dotenv()

    # Load data
    df = pl.read_csv("data/HeliusData.csv")

    # Process a small batch as test
    batch_results = process_batch(df, 0, 5)

    # Print summary
    print(f"\nProcessed {len(batch_results.results)} transactions")
    print(f"Success rate: {batch_results.success_rate:.2%}")
    print(f"Successful classifications: {batch_results.successful_classifications}")
    print(f"Failed classifications: {batch_results.failed_classifications}")

    # Save results to JSON
    batch_results.to_json("classification_results.json")

    # Print detailed results
    for result in batch_results.results:
        print(f"\nTransaction {result.index}:")
        if result.classification:
            print(f"Classification: {result.classification}")
        else:
            print(f"Error: {result.error}")

# %%
