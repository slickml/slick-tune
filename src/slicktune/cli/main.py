"""Click CLI for slick-tune."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from slicktune.objectives import SFTObjective
from slicktune.recipes import load_trained, run_probes
from slicktune.strategies import FullStrategy, LoRAStrategy, QLoRAStrategy
from slicktune.tuner import Tuner

console = Console()


def _strategy_from_name(name: str) -> LoRAStrategy | QLoRAStrategy | FullStrategy:
    """Map a CLI strategy name to a strategy instance.

    Parameters
    ----------
    name : str
        One of ``lora``, ``qlora``, ``full``.

    Returns
    -------
    LoRAStrategy or QLoRAStrategy or FullStrategy
        Strategy instance.

    Raises
    ------
    click.BadParameter
        If ``name`` is unknown.
    """
    mapping: dict[str, LoRAStrategy | QLoRAStrategy | FullStrategy] = {
        "lora": LoRAStrategy(),
        "qlora": QLoRAStrategy(),
        "full": FullStrategy(),
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise click.BadParameter(f"Unknown strategy: {name}") from exc


@click.group()
def cli() -> None:
    """slick-tune: composable LLM fine-tuning."""


@cli.command("train")
@click.option(
    "--model",
    "model_id",
    default="HuggingFaceTB/SmolLM2-135M-Instruct",
    show_default=True,
    help="Base Hugging Face model id.",
)
@click.option(
    "--strategy",
    type=click.Choice(["lora", "qlora", "full"], case_sensitive=False),
    default="lora",
    show_default=True,
)
@click.option(
    "--data",
    "data_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="SFT JSONL path.",
)
@click.option(
    "--output",
    "output_dir",
    type=click.Path(path_type=Path),
    default=Path("outputs/run"),
    show_default=True,
)
@click.option("--epochs", default=20.0, show_default=True, type=float)
@click.option("--lr", default=2e-4, show_default=True, type=float)
@click.option("--max-seq-length", default=512, show_default=True, type=int)
@click.option("--batch-size", default=1, show_default=True, type=int)
@click.option("--grad-accum", default=4, show_default=True, type=int)
def train(
    model_id: str,
    strategy: str,
    data_path: Path,
    output_dir: Path,
    epochs: float,
    lr: float,
    max_seq_length: int,
    batch_size: int,
    grad_accum: int,
) -> None:
    """Fine-tune with SFT + the chosen parameter strategy."""
    console.print(f"[bold]Training[/bold] strategy={strategy} model={model_id} data={data_path}")
    tuner = Tuner(
        model_id=model_id,
        strategy=_strategy_from_name(strategy.lower()),
        objective=SFTObjective(),
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        max_seq_length=max_seq_length,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
    )
    result = tuner.fit(data_path)
    m = result.metrics
    console.print(f"[green]Saved[/green] → {result.output_dir}")
    if m.trainable_percent is not None:
        console.print(
            f"train_loss={m.train_loss} trainable={m.trainable_params}/{m.total_params} "
            f"({m.trainable_percent:.2f}%)"
        )
    else:
        console.print(f"train_loss={m.train_loss}")


@cli.command("probe")
@click.option(
    "--model-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory from `slick-tune train`.",
)
@click.option(
    "--probes",
    "probe_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Probe JSONL with prompt + must_contain.",
)
@click.option("--max-new-tokens", default=128, show_default=True, type=int)
def probe(model_dir: Path, probe_path: Path, max_new_tokens: int) -> None:
    """Ask probe questions to verify personal facts were learned."""
    model, tokenizer = load_trained(model_dir)
    report = run_probes(
        model,
        tokenizer,
        probe_path,
        max_new_tokens=max_new_tokens,
    )
    table = Table(title=f"Probe pass rate: {report.pass_rate:.0%}")
    table.add_column("Pass")
    table.add_column("Prompt")
    table.add_column("Expected")
    table.add_column("Generation")
    for item in report.results:
        table.add_row(
            "✓" if item.passed else "✗",
            item.prompt,
            item.must_contain,
            item.generation[:160],
        )
    console.print(table)
    if report.pass_rate < 1.0:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    cli()
