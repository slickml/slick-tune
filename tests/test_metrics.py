"""Tests for metrics tracking."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from assertpy import assert_that

from slicktune.metrics import MetricsTracker, TrainingMetrics, count_parameters


def test_metrics_tracker_roundtrip(tmp_path: Path) -> None:
    """Save and reload training metrics."""
    metrics = TrainingMetrics(
        strategy="lora",
        objective="sft",
        model_id="tiny",
        train_loss=1.23,
        trainable_params=100,
        total_params=1000,
    )
    tracker = MetricsTracker(tmp_path)
    path = tracker.save(metrics)
    assert_that(path.exists()).is_true()

    loaded = tracker.load()
    assert_that(loaded.train_loss).is_equal_to(1.23)
    assert_that(loaded.trainable_percent).is_equal_to(10.0)


def test_trainable_percent_none_without_counts() -> None:
    """Percent is None when parameter counts are missing."""
    metrics = TrainingMetrics(strategy="lora", objective="sft", model_id="tiny")
    assert_that(metrics.trainable_percent).is_none()


def test_trainable_percent_none_when_total_zero() -> None:
    """Percent is None when total parameter count is zero."""
    metrics = TrainingMetrics(
        strategy="lora",
        objective="sft",
        model_id="tiny",
        trainable_params=0,
        total_params=0,
    )
    assert_that(metrics.trainable_percent).is_none()


def test_metrics_tracker_load_missing(tmp_path: Path) -> None:
    """Load raises when metrics.json is absent."""
    tracker = MetricsTracker(tmp_path)
    with pytest.raises(FileNotFoundError, match="No metrics"):
        tracker.load()


def test_count_parameters() -> None:
    """Count trainable vs total parameters on a tiny module."""
    linear = torch.nn.Linear(2, 2)
    linear.weight.requires_grad = True
    linear.bias.requires_grad = False
    trainable, total = count_parameters(linear)
    assert_that(trainable).is_equal_to(linear.weight.numel())
    assert_that(total).is_equal_to(linear.weight.numel() + linear.bias.numel())


def test_count_parameters_accepts_simple_namespace_model() -> None:
    """count_parameters works with objects exposing parameters()."""
    p_train = torch.nn.Parameter(torch.ones(3), requires_grad=True)
    p_frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    model = SimpleNamespace(parameters=lambda: iter([p_train, p_frozen]))
    trainable, total = count_parameters(model)
    assert_that(trainable).is_equal_to(3)
    assert_that(total).is_equal_to(5)
