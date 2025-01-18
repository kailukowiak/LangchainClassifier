# %%
import matplotlib.pyplot as plt
import polars as pl

df = pl.read_csv("out_data/batch_classification_comparison.csv")

# %%


# %%
# Show what percent of errors are their in the data
df.select("error").null_count()
# %%
pivot_df = (
    df.select(["actual_beam_tag", "correct_tag"])
    .to_pandas()
    .pivot_table(
        index="actual_beam_tag",
        columns="correct_tag",
        values="correct_tag",
        aggfunc="size",
        fill_value=0,
    )
)

# Calculate percentages
pivot_df_pct = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100


# %%
# Set style
plt.style.use("dark_background")

# Create figure and axis with dark background
plt.figure(
    figsize=(12, 6), facecolor="#1a1a1a"
)  # Made figure wider to accommodate legend
ax = plt.gca()
ax.set_facecolor("#1a1a1a")

# Create plot with new colors
ax = pivot_df_pct.plot(
    kind="bar", stacked=True, color=["#D05121", "#404040"], width=0.8
)

# Remove all grid lines
ax.grid(False)

# Remove top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#404040")
ax.spines["bottom"].set_color("#404040")

# Customize title and labels
plt.title("Classification Accuracy by Tag", color="#FFFFFF", pad=20, fontsize=12)
plt.xlabel("Tag", color="#FFFFFF", labelpad=10)
plt.ylabel("Percentage", color="#FFFFFF", labelpad=10)

# Position legend outside the plot
plt.legend(
    title="Classification",
    labels=["Incorrect", "Correct"],
    facecolor="#1a1a1a",
    labelcolor="white",
    edgecolor="#404040",
    bbox_to_anchor=(1.05, 1),  # Position legend outside
    loc="upper left",
)

# Customize ticks
plt.xticks(rotation=45, color="#FFFFFF", ha="right")
plt.yticks(color="#FFFFFF")

# Add percentage labels
for c in ax.containers:
    ax.bar_label(c, fmt="%.0f%%", color="#FFFFFF", label_type="center")

# Adjust layout to ensure legend fits
plt.tight_layout()

plt.show()


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
        percentage=pl.col("count") / pl.col("total") * 100
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
