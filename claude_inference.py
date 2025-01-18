# %%

import os

import polars as pl
from dotenv import load_dotenv

from langchainclassifier import BatchProcessor, create_comparison_df

# %%

if __name__ == "__main__":
    # main()
    load_dotenv()
    data_path = os.getenv("TRANSACTIONS_DATA_PATH")
    df = pl.read_csv(data_path)
    df = df.filter(
        pl.col("Beam Tag").is_not_null() & pl.col("Beam Label").is_not_null()
    )

    df = df.group_by(["Beam Tag", "Beam Label"]).head(100)
    # randomly sort the data
    df = df.sample(5)
    # df = df.head(50)

    # Initialize processor with desired parameters
    processor = BatchProcessor(
        model_name="gemini-pro", batch_size=25, max_retries=3, df=df
    )
    # processor = BatchProcessor(
    #     model_name="claude-3-5-sonnet-latest", batch_size=25, max_retries=3, df=df
    # )

    # Process dataset
    results = processor.process_dataset(df)

    # Save results
    results.to_json("out_data/batch_classification_results.json")

    # Create and save comparison DataFrame
    comparison_df = create_comparison_df(results, df)
    comparison_df.write_csv("out_data/batch_classification_comparison.csv")
# %%
