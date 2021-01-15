import time
from rich.progress import (
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    Progress,
    ProgressColumn,
    Text
)

class TrainingExamples(ProgressColumn):
    """Renders human readable transfer speed."""

    def render(self, task: "Task") -> Text:
        """Show data transfer speed."""
        speed = task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        #data_speed = filesize.decimal(int(speed))
        it_speed = int(speed) * int(task.fields['batch_size'])
        return Text(f"{it_speed} Examples/sec", style="progress.data.speed")

class TrainingProgress:
    def __init__(self):
        self.bar = Progress(
            TextColumn("[bold blue]{task.fields[mode]} {task.fields[epoch]}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            #"•",
            #TextColumn("[bold blue]{task.fields[loss]}", justify="right"),
            "•",
            TrainingExamples(),
            "•",
            TimeRemainingColumn(),
        )
    
    def close(self):
        self.bar.stop()
    
    def __enter__(self):
        return self.bar

    def __exit__(self, *_):
        self.close()