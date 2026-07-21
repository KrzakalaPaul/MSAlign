from preprocessing import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for MSAlign training")

    parser.add_argument("--dataset_name",              type=str,   default="massspecgym",
                        choices=["massspecgym", "spectraverse"])
    parser.add_argument("--split_method",              type=str,   default="formula",
                        choices=["formula", "random", "as_provided"])
    parser.add_argument("--overwrite",                 action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for seed in [1,2,3,4,5]:
            
        split(
            dataset_name=args.dataset_name,
            split_method=args.split_method,
            overwrite=args.overwrite,
            seed=seed,
            add_seed_name=True,
        )

