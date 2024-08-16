import config
import factory
import argparse

def model_list():
    return [model["name"] for model in config.models]

if __name__ == "__main__":
    models = model_list()
    parser = argparse.ArgumentParser(description="Train arguments")
    parser.add_argument("-n", "--epoch_num", type=int, help="Train Epoch Number", default=10)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch Size", default=64)
    parser.add_argument("-l", "--learning_rate", type=float, help="Learning Rate", default=1e-3)
    parser.add_argument(
        "-w",
        "--window_sizes",
        type=int,
        nargs="+",
        help="Window Sizes",
        default=[3, 6, 12, 24, 36, 72, 144],
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        nargs="+",
        choices=models,
        default=models,
        help="Model name to train.",
    )
    args = parser.parse_args()
