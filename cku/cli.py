import argparse
import json

from . import __version__


def make_parser():
    parser = argparse.ArgumentParser(prog="cku", description="Constrained Knowledge Unlearning")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ["score", "train"]:
        sub = commands.add_parser(command)
        sub.add_argument("--config", help="JSON config; explicit CLI flags override config values")
        for name in ["model", "data", "output", "revision"]:
            sub.add_argument(f"--{name}")
        sub.add_argument("--device", help="auto, cpu, cuda or cuda:N (one device/process)")
        sub.add_argument("--dtype", choices=["float32", "bfloat16"])
        sub.add_argument("--prompt-format", choices=["plain", "chat"])
        sub.add_argument("--max-length", type=int)
        sub.add_argument("--seed", type=int)
        if command == "score":
            sub.add_argument("--layer-start", type=int, help="Zero-based, inclusive; default 8")
            sub.add_argument("--layer-end", type=int, help="Zero-based, exclusive; default 12")
            sub.add_argument("--num-samples", type=int)
        else:
            sub.add_argument("--masks")
            sub.add_argument("--retain-data")
            for name in ["learning-rate", "harmful-threshold", "kl-weight", "max-grad-norm"]:
                sub.add_argument(f"--{name}", type=float)
            for name in ["batch-size", "gradient-accumulation-steps", "epochs", "max-steps"]:
                sub.add_argument(f"--{name}", type=int)
            sub.add_argument(
                "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None
            )
    masks = commands.add_parser("mask", help="Build a fixed mask from importance scores")
    masks.add_argument("--scores", required=True)
    masks.add_argument("--output", required=True)
    masks.add_argument("--lock-ratio", type=float, default=0.8)
    return parser


def main(argv=None):
    parser = make_parser()
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command in {"score", "train"}:
            from .config import resolve_options

            config_path = args.pop("config")
            options = resolve_options(command, args, config_path)
            if command == "score":
                from .scoring import score_file

                result = score_file(options)
            else:
                from .training import train_file

                result = train_file(options)
        elif command == "mask":
            from .artifacts import build_masks

            result = build_masks(args["scores"], args["output"], args["lock_ratio"])
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"cku: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
