# %%
import time
from datetime import datetime

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def simulate_processing(n_transactions=10):
    # Create columns for main progress bar (with time)
    progress_columns = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    ]

    # Create columns for accuracy metrics (without time)
    accuracy_columns = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ]

    start_time = datetime.now()

    # Create a single Progress instance with all tasks
    with Progress(*progress_columns) as progress:
        # Create progress bars
        task_overall = progress.add_task(
            "[cyan]Processing transactions...", total=n_transactions
        )

        # Add accuracy tasks with different columns
        task_label_accuracy = progress.add_task(
            "[green]Beam Label Accuracy",
            total=100,
            columns=accuracy_columns,  # Use custom columns without time
        )

        task_tag_accuracy = progress.add_task(
            "[yellow]Beam Tag Accuracy",
            total=100,
            columns=accuracy_columns,  # Use custom columns without time
        )

        # Simulate processing
        for i in range(n_transactions):
            time.sleep(0.5)  # Simulate work
            progress.update(task_overall, advance=1)

            # Update accuracies (simulated values)
            label_accuracy = min(100, (i + 1) * 10)
            tag_accuracy = min(80, (i + 1) * 8)

            progress.update(task_label_accuracy, completed=label_accuracy)
            progress.update(task_tag_accuracy, completed=tag_accuracy)

    # Calculate and display total elapsed time
    elapsed_time = datetime.now() - start_time
    console = Console()
    console.print(
        f"\n[cyan]Total elapsed time: {elapsed_time.total_seconds():.2f} seconds"
    )


if __name__ == "__main__":
    simulate_processing(10)

# %%
