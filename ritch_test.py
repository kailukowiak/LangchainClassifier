# %%
import time

from rich import print
from rich.panel import Panel
from rich.progress import Progress, track

for i in track(range(20), description="Processing..."):
    time.sleep(1)  # Simulate work being done
# %%


with Progress() as progress:
    task1 = progress.add_task("[red]Downloading...", total=1000)
    task2 = progress.add_task("[green]Processing...", total=1000)
    task3 = progress.add_task("[cyan]Cooking...", total=1000)

    while not progress.finished:
        progress.update(task1, advance=0.5)
        progress.update(task2, advance=0.3)
        progress.update(task3, advance=0.9)
        # print(Panel(f"Hello,\n\n [red]World!{progress}", title="Test"))
        time.sleep(0.02)
# %%

print(Panel("Hello,\n\n [red]World!"))
# %%
