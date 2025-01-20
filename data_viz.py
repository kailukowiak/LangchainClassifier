# %%
import matplotlib.pyplot as plt
import polars as pl

df = pl.read_csv("out_data/ModelName.GEMINI_15_8Bbatch_classification_comparison.csv")
# drop if column `error` is not null
# df = df.filter(pl.col("error").is_null())
# filter actual_beam_label is not null or actual_beam_tag is not null
df = df.filter(
    pl.col("actual_beam_label").is_not_null() & pl.col("actual_beam_tag").is_not_null()
)
# %%


# Show what percent of errors are their in the data
df.select("error").null_count()
# %%


# %%
def create_classification_chart(
    df: pl.DataFrame,
    grouping_col: str,
    correct_col: str = "correct_tag",
    figsize: tuple = (12, 6),
) -> None:
    """
    Create a stacked bar chart showing classification accuracy using only Polars.

    Args:
        df: Polars DataFrame containing the classification data
        grouping_col: Column name to group by (e.g., 'actual_beam_tag' or 'Beam Label')
        correct_col: Column name indicating correct/incorrect classification
        figsize: Tuple of (width, height) for the plot
    """
    # Calculate counts and percentages using only Polars
    results = (
        df.group_by([grouping_col, correct_col])
        .agg(pl.len().alias("count"))
        .with_columns(pl.col("count").cast(pl.Float64))
    )

    # Calculate total counts per group
    totals = results.group_by(grouping_col).agg(pl.col("count").sum().alias("total"))

    # Join and calculate percentages
    results = results.join(totals, on=grouping_col).with_columns(
        percentage=pl.col("count") / pl.col("total") * 100,
    )

    # Pivot the results
    pivot_df = results.pivot(
        values="percentage",
        index=grouping_col,
        on=correct_col,
        aggregate_function="sum",
    ).fill_null(0)

    # Get column names
    columns = pivot_df.columns
    value_columns = [col for col in columns if col != grouping_col]

    # Create plot
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=figsize, facecolor="#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    # Get the data in the right format for plotting
    categories = pivot_df.get_column(grouping_col).to_list()
    incorrect = pivot_df.get_column(value_columns[0]).to_list()
    correct = pivot_df.get_column(value_columns[1]).to_list()

    # Create stacked bars
    x = range(len(categories))
    ax.bar(x, incorrect, color="#D05121", label="Incorrect", width=0.8)
    ax.bar(x, correct, bottom=incorrect, color="#404040", label="Correct", width=0.8)

    # Style customization
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#404040")
    ax.spines["bottom"].set_color("#404040")

    # Labels and title
    ax.set_title(
        f"Classification Accuracy by {grouping_col}",
        color="#FFFFFF",
        pad=20,
        fontsize=12,
    )
    ax.set_xlabel(grouping_col, color="#FFFFFF", labelpad=10)
    ax.set_ylabel("Percentage", color="#FFFFFF", labelpad=10)

    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", color="#FFFFFF")

    # Set y-axis labels
    ax.tick_params(axis="y", colors="#FFFFFF")

    # Add percentage labels
    for i in range(len(x)):
        # Label for incorrect portion
        if incorrect[i] > 0:
            ax.text(
                x[i],
                incorrect[i] / 2,
                f"{int(incorrect[i])}%",
                ha="center",
                va="center",
                color="#FFFFFF",
            )
        # Label for correct portion
        if correct[i] > 0:
            ax.text(
                x[i],
                incorrect[i] + correct[i] / 2,
                f"{int(correct[i])}%",
                ha="center",
                va="center",
                color="#FFFFFF",
            )

    # Legend
    ax.legend(
        title="Classification",
        labels=["Correct", "Incorrect"],
        facecolor="#1a1a1a",
        labelcolor="white",
        edgecolor="#404040",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    plt.tight_layout()
    plt.show()


# For beam tag visualization
# %%
create_classification_chart(df, grouping_col="actual_beam_tag")

# For beam label visualization
# %%

create_classification_chart(
    df, grouping_col="actual_beam_label", correct_col="correct_label"
)
# Usage examples:
# For beam tag chart
# create_classification_chart(df, grouping_col="actual_beam_tag")

# For beam label chart
# create_classification_chart(df, grouping_col="Beam Label")

# %%
