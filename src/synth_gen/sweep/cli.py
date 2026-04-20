"""Argument-parsing helpers shared by sweep drivers.

Keeps CTGAN option wiring and integer-list parsing in one place so the two
drivers stay in lockstep.
"""

from __future__ import annotations

import argparse

from .config import (
    CTGAN_DEFAULT_BATCH_SIZE,
    CTGAN_DEFAULT_DECAY,
    CTGAN_DEFAULT_DISCRIMINATOR_DIM,
    CTGAN_DEFAULT_EMBEDDING_DIM,
    CTGAN_DEFAULT_EPOCHS,
    CTGAN_DEFAULT_GENERATOR_DIM,
    CTGAN_DEFAULT_LR,
    CTGAN_DEFAULT_PAC,
)


def parse_int_list(arg: str) -> list[int]:
    """Parse a comma-separated positive-integer list for argparse."""
    vals = [v.strip() for v in arg.split(",") if v.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("Expected at least one integer value")
    try:
        parsed = [int(v) for v in vals]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer list: {arg}") from exc
    if any(v <= 0 for v in parsed):
        raise argparse.ArgumentTypeError("All values must be positive integers")
    return parsed


def parse_int_tuple(arg: str) -> tuple[int, ...]:
    """Parse a comma-separated positive-integer tuple for argparse."""
    return tuple(parse_int_list(arg))


def add_ctgan_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach shared CTGAN tuning flags to ``parser``."""
    parser.add_argument(
        "--ctgan-epochs",
        type=int,
        default=CTGAN_DEFAULT_EPOCHS,
        help=f"CTGAN epochs (default: {CTGAN_DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--ctgan-batch-size",
        type=int,
        default=CTGAN_DEFAULT_BATCH_SIZE,
        help=f"CTGAN batch size (default: {CTGAN_DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--ctgan-generator-dim",
        type=parse_int_tuple,
        default=CTGAN_DEFAULT_GENERATOR_DIM,
        help=(
            "CTGAN generator dims (comma-separated ints; default: "
            f"{','.join(str(v) for v in CTGAN_DEFAULT_GENERATOR_DIM)})"
        ),
    )
    parser.add_argument(
        "--ctgan-discriminator-dim",
        type=parse_int_tuple,
        default=CTGAN_DEFAULT_DISCRIMINATOR_DIM,
        help=(
            "CTGAN discriminator dims (comma-separated ints; default: "
            f"{','.join(str(v) for v in CTGAN_DEFAULT_DISCRIMINATOR_DIM)})"
        ),
    )
    parser.add_argument(
        "--ctgan-embedding-dim",
        type=int,
        default=CTGAN_DEFAULT_EMBEDDING_DIM,
        help=f"CTGAN embedding dim (default: {CTGAN_DEFAULT_EMBEDDING_DIM})",
    )
    parser.add_argument(
        "--ctgan-lr",
        type=float,
        default=CTGAN_DEFAULT_LR,
        help=f"CTGAN learning rate (default: {CTGAN_DEFAULT_LR})",
    )
    parser.add_argument(
        "--ctgan-decay",
        type=float,
        default=CTGAN_DEFAULT_DECAY,
        help=f"CTGAN weight decay (default: {CTGAN_DEFAULT_DECAY})",
    )
    parser.add_argument(
        "--ctgan-pac",
        type=int,
        default=CTGAN_DEFAULT_PAC,
        help=f"CTGAN PAC (default: {CTGAN_DEFAULT_PAC})",
    )
    parser.add_argument(
        "--ctgan-verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CTGAN training/sampling progress logs (default: enabled)",
    )


def ctgan_params_from_args(args: argparse.Namespace) -> dict:
    """Build the ``ctgan_params`` kwargs dict from parsed arguments."""
    return {
        "epochs": int(args.ctgan_epochs),
        "batch_size": int(args.ctgan_batch_size),
        "generator_dim": tuple(int(v) for v in args.ctgan_generator_dim),
        "discriminator_dim": tuple(int(v) for v in args.ctgan_discriminator_dim),
        "embedding_dim": int(args.ctgan_embedding_dim),
        "generator_lr": float(args.ctgan_lr),
        "discriminator_lr": float(args.ctgan_lr),
        "generator_decay": float(args.ctgan_decay),
        "discriminator_decay": float(args.ctgan_decay),
        "pac": int(args.ctgan_pac),
        "log_frequency": True,
        "verbose": bool(args.ctgan_verbose),
    }


def format_ctgan_params(params: dict) -> str:
    """Stable one-line summary used by both drivers."""
    return (
        f"epochs={params['epochs']}, batch_size={params['batch_size']}, "
        f"generator_dim={params['generator_dim']}, "
        f"discriminator_dim={params['discriminator_dim']}, "
        f"embedding_dim={params['embedding_dim']}, "
        f"lr={params['generator_lr']}, decay={params['generator_decay']}, "
        f"pac={params['pac']}, verbose={params['verbose']}"
    )


__all__ = [
    "parse_int_list",
    "parse_int_tuple",
    "add_ctgan_arguments",
    "ctgan_params_from_args",
    "format_ctgan_params",
]
