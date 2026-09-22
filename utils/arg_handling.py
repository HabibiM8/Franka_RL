import argparse
import functools

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-en",
        "--task",
        help="Task name.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="Seed of the experiment.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-f",
        "--features",
        nargs="*",
        help="List of features for the Q-networks.",
        type=int,
        default=[256, 256, 256],
    )
    parser.add_argument(
        "-b_s",
        "--buffer_size",
        help="Buffer size.",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        help="Batch size for training.",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "-gamma",
        "--gamma",
        help="Discounting factor.",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "-tau",
        "--tau",
        help="Soft update coefficient.",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="Learning rate.",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "-ns",
        "--n_steps",
        help="Number of steps to perform.",
        type=int,
        default=15_000,
    )
    parser.add_argument(
        "-ts",
        "--learning_starts",
        help="Number of initial samples before the training starts.",
        type=int,
        default=10_000,
    )
    return parser

def parse_args(argvs=None) -> argparse.Namespace:
    return build_parser().parse_args(argvs)

def with_args(func):
    @functools.wraps(func)
    def wrapper(argvs=None):
        args = parse_args(argvs)
        return func(args)
    return wrapper



