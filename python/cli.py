#!/usr/bin/ven python
import typer
from pathlib import Path

import mt_code
from mt_code.config import Config

app = typer.Typer()


@app.command()
def train(output_path: Path = typer.Argument(None, help="Directory to save model in ")):
    config = Config()
    mt_code.train(output_path, config)


@app.command()
def evaluate(model_path: Path = typer.Argument(None, help="Path of trained model"),
             output_path: Path = typer.Argument(None, help="Output directory of evaluation results")):
    pass


if __name__ == "__main__":
    app()
