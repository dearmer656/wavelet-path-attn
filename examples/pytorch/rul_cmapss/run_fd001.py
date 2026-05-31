import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/cmapss")
    parser.add_argument("--output_dir", type=str, default="examples/pytorch/rul_cmapss/outputs/fd001")
    parser.add_argument(
        "--model",
        type=str,
        choices=["path", "rope", "alibi", "nope", "lstm", "tcn"],
        default="path",
    )
    parser.add_argument("--window_size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_rul", type=int, default=125)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--head_dim", type=int, default=16)
    parser.add_argument("--max_position_embeddings", type=int, default=256)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--critical_rul", type=int, default=30)
    parser.add_argument("--include_phm", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    from train_eval import run_experiment
    run_experiment(args)


if __name__ == "__main__":
    main()
