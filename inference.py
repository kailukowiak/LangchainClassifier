# %%
from enum import Enum
from typing import Optional

import polars as pl
import typer
from dotenv import load_dotenv

from langchainclassifier import BatchProcessor, create_comparison_df

load_dotenv()


# %%
class ModelName(str, Enum):
    HAIKU = "claude-3-5-haiku-20241022"
    SONNET = "claude-3-5-sonnet-20241022"
    GEMINI_2 = "gemini-2.0-flash-exp"
    GEMINI_15 = "gemini-1.5-flash"
    GEMINI_15_8B = "gemini-1.5-flash-8b"


app = typer.Typer(name="crypto-classifier", add_completion=False)


@app.command(help="Classify a CSV file with the specified model")
def main(
    model_name: ModelName = typer.Option(
        ModelName.HAIKU,
        "--model",
        "-m",
        help="The name of the model to use for classification",
    ),
    data_path: Optional[str] = typer.Option(
        "data/HeliusData.csv",
        "--path",
        "-p",
        help="The path to the data file",
    ),
    n_rows: Optional[int] = typer.Option(
        10,
        "--n-rows",
        "-n",
        help="The number of rows to process",
    ),
    batch_size: Optional[int] = typer.Option(
        25, "--batch-size", "-b", help="The batch size to use"
    ),
):
    df = pl.read_csv(data_path)
    df = df.filter(
        pl.col("Beam Tag").is_not_null() & pl.col("Beam Label").is_not_null()
    )

    df = df.group_by(["Beam Tag", "Beam Label"]).head(n_rows)
    # randomly sort the data
    df = df.sample(n_rows)
    # df = df.head(50)

    # Initialize processor with desired parameters
    processor = BatchProcessor(
        model_name=model_name, batch_size=batch_size, max_retries=3, df=df
    )
    # processor = BatchProcessor(
    #     model_name="claude-3-5-sonnet-latest", batch_size=25, max_retries=3, df=df
    # )

    # Process dataset
    results = processor.process_dataset(df)

    # Save results
    results.to_json(f"out_data/{model_name}batch_classification_results.json")

    # Create and save comparison DataFrame
    comparison_df = create_comparison_df(results, df)
    comparison_df.write_csv(f"out_data/{model_name}batch_classification_comparison.csv")


if __name__ == "__main__":
    app()

# %%
