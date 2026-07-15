"""Click CLI for slicktune."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from slicktune import __version__
from slicktune.eval import LLMJudge, SubstringJudge, compute_holdout_perplexity, run_judge_on_probes
from slicktune.objectives import DPOObjective, KTOObjective, ORPOObjective, SFTObjective
from slicktune.recipes import load_trained, run_probes
from slicktune.strategies import (
    AdaLoRAStrategy,
    DoRAStrategy,
    FullStrategy,
    LoRAStrategy,
    QLoRAStrategy,
)
from slicktune.tuner import Tuner
from slicktune.types import Objective

console = Console()

StrategyName = LoRAStrategy | DoRAStrategy | AdaLoRAStrategy | QLoRAStrategy | FullStrategy


def _strategy_from_name(name: str) -> StrategyName:
    """Map a CLI strategy name to a strategy instance.

    Parameters
    ----------
    name : str
        One of ``lora``, ``dora``, ``adalora``, ``qlora``, ``full``.

    Returns
    -------
    StrategyName
        Strategy instance.

    Raises
    ------
    click.BadParameter
        If ``name`` is unknown.
    """
    mapping: dict[str, StrategyName] = {
        "lora": LoRAStrategy(),
        "dora": DoRAStrategy(),
        "adalora": AdaLoRAStrategy(),
        "qlora": QLoRAStrategy(),
        "full": FullStrategy(),
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise click.BadParameter(f"Unknown strategy: {name}") from exc


def _objective_from_name(name: str, *, beta: float) -> Objective:
    """Map a CLI objective name to an objective instance.

    Parameters
    ----------
    name : str
        One of ``sft``, ``dpo``, ``orpo``, ``kto``.
    beta : float
        Preference KL / odds-ratio coefficient (ignored for SFT).

    Returns
    -------
    Objective
        Objective instance.

    Raises
    ------
    click.BadParameter
        If ``name`` is unknown.
    """
    mapping: dict[str, Objective] = {
        "sft": SFTObjective(),
        "dpo": DPOObjective(beta=beta),
        "orpo": ORPOObjective(beta=beta),
        "kto": KTOObjective(beta=beta),
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise click.BadParameter(f"Unknown objective: {name}") from exc


@click.group()
@click.version_option(version=__version__, prog_name="slicktune")
def cli() -> None:
    """slicktune: composable LLM fine-tuning."""


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
    type=click.Choice(["lora", "dora", "adalora", "qlora", "full"], case_sensitive=False),
    default="lora",
    show_default=True,
)
@click.option(
    "--objective",
    "objective_name",
    type=click.Choice(["sft", "dpo", "orpo", "kto"], case_sensitive=False),
    default="sft",
    show_default=True,
)
@click.option(
    "--beta",
    default=0.1,
    show_default=True,
    type=float,
    help="Preference beta (DPO / ORPO / KTO).",
)
@click.option(
    "--data",
    "data_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Training JSONL path (SFT messages, DPO/ORPO prefs, or KTO rows).",
)
@click.option(
    "--eval-data",
    "eval_data",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional holdout SFT JSONL for perplexity after training.",
)
@click.option(
    "--probes",
    "probe_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional probe JSONL judged after training.",
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
@click.option(
    "--batch-size",
    default=1,
    show_default=True,
    type=int,
    help="Per-device train batch size (KTO auto-bumps to at least 2).",
)
@click.option("--grad-accum", default=4, show_default=True, type=int)
def train(
    model_id: str,
    strategy: str,
    objective_name: str,
    beta: float,
    data_path: Path,
    eval_data: Path | None,
    probe_path: Path | None,
    output_dir: Path,
    epochs: float,
    lr: float,
    max_seq_length: int,
    batch_size: int,
    grad_accum: int,
) -> None:
    """Fine-tune with the chosen objective + parameter strategy."""
    objective = _objective_from_name(objective_name.lower(), beta=beta)
    console.print(
        f"[bold]Training[/bold] objective={objective.name} strategy={strategy} "
        f"model={model_id} data={data_path}"
    )
    tuner = Tuner(
        model_id=model_id,
        strategy=_strategy_from_name(strategy.lower()),
        objective=objective,
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        max_seq_length=max_seq_length,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        eval_data=eval_data,
        probe_path=probe_path,
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
    if m.eval_perplexity is not None:
        console.print(f"eval_loss={m.eval_loss} perplexity={m.eval_perplexity:.3f}")
    if m.judge_score is not None:
        console.print(f"judge_score={m.judge_score:.2%}")


@cli.command("probe")
@click.option(
    "--model-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory from `slicktune train`.",
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
        model=model,
        tokenizer=tokenizer,
        probe_path=probe_path,
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


@cli.command("eval")
@click.option(
    "--model-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory from `slicktune train`.",
)
@click.option(
    "--eval-data",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Holdout SFT JSONL for perplexity.",
)
@click.option(
    "--probes",
    "probe_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Probe JSONL for judge scoring.",
)
@click.option(
    "--judge",
    "judge_name",
    type=click.Choice(["substring", "llm"], case_sensitive=False),
    default="substring",
    show_default=True,
)
@click.option("--max-seq-length", default=512, show_default=True, type=int)
@click.option("--max-new-tokens", default=128, show_default=True, type=int)
def eval_cmd(
    model_dir: Path,
    eval_data: Path | None,
    probe_path: Path | None,
    judge_name: str,
    max_seq_length: int,
    max_new_tokens: int,
) -> None:
    """Run holdout perplexity and/or probe judging on a saved checkpoint."""
    if eval_data is None and probe_path is None:
        raise click.UsageError("Provide --eval-data and/or --probes")

    model, tokenizer = load_trained(model_dir)
    if eval_data is not None:
        holdout = compute_holdout_perplexity(
            model=model,
            tokenizer=tokenizer,
            data=eval_data,
            max_length=max_seq_length,
        )
        console.print(
            f"[bold]Holdout[/bold] loss={holdout.eval_loss:.4f} "
            f"perplexity={holdout.perplexity:.3f} n={holdout.num_examples}"
        )

    if probe_path is not None:
        if judge_name.lower() == "llm":
            judge: SubstringJudge | LLMJudge = LLMJudge(model=model, tokenizer=tokenizer)
        else:
            judge = SubstringJudge()
        report = run_judge_on_probes(
            model=model,
            tokenizer=tokenizer,
            probe_path=probe_path,
            judge=judge,
            max_new_tokens=max_new_tokens,
        )
        table = Table(title=f"Judge mean score: {report.mean_score:.0%}")
        table.add_column("Score")
        table.add_column("Prompt")
        table.add_column("Rationale")
        for item in report.results:
            table.add_row(f"{item.score:.2f}", item.prompt, item.rationale[:120])
        console.print(table)


if __name__ == "__main__":  # pragma: no cover
    cli()
