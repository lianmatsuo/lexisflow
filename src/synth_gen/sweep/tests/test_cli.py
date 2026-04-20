"""Tests for shared CLI helpers."""

from __future__ import annotations

import argparse

import pytest

from synth_gen.sweep.cli import (
    add_ctgan_arguments,
    ctgan_params_from_args,
    format_ctgan_params,
    parse_int_list,
    parse_int_tuple,
)


def test_parse_int_list_valid():
    assert parse_int_list("1,3,5") == [1, 3, 5]
    assert parse_int_list("7") == [7]


def test_parse_int_list_rejects_non_positive():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_int_list("0,1")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_int_list("-2")


def test_parse_int_list_rejects_empty():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_int_list("")


def test_parse_int_tuple_returns_tuple():
    result = parse_int_tuple("2,4,6")
    assert isinstance(result, tuple)
    assert result == (2, 4, 6)


def test_ctgan_params_builds_consistent_dict():
    parser = argparse.ArgumentParser()
    add_ctgan_arguments(parser)
    args = parser.parse_args([])
    params = ctgan_params_from_args(args)
    # Defaults propagate and generator/discriminator dims match
    assert params["epochs"] > 0
    assert isinstance(params["generator_dim"], tuple)
    assert isinstance(params["discriminator_dim"], tuple)
    # Shared LR and decay for generator and discriminator
    assert params["generator_lr"] == params["discriminator_lr"]
    assert params["generator_decay"] == params["discriminator_decay"]
    assert params["log_frequency"] is True
    # Summary is one line and references every critical knob
    line = format_ctgan_params(params)
    assert "epochs" in line and "pac" in line
    assert "\n" not in line


def test_ctgan_params_respects_overrides():
    parser = argparse.ArgumentParser()
    add_ctgan_arguments(parser)
    args = parser.parse_args(
        [
            "--ctgan-epochs",
            "10",
            "--ctgan-batch-size",
            "64",
            "--ctgan-generator-dim",
            "32,32",
            "--ctgan-discriminator-dim",
            "16,16",
            "--no-ctgan-verbose",
        ]
    )
    params = ctgan_params_from_args(args)
    assert params["epochs"] == 10
    assert params["batch_size"] == 64
    assert params["generator_dim"] == (32, 32)
    assert params["discriminator_dim"] == (16, 16)
    assert params["verbose"] is False
