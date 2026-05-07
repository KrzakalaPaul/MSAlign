from preprocessing import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for MSAlign training")

    parser.add_argument("--dataset_name",              type=str,   default="massspecgym",
                        choices=["massspecgym", "spectraverse"])
    parser.add_argument("--split_method",              type=str,   default="formula",
                        choices=["formula", "random", "scaffold"])
    parser.add_argument("--n_candidates",              type=int,   default=256)
    parser.add_argument("--sources",                   type=str,   nargs="+", default=["1M", "4M", "118M"],
                        choices=["1M", "4M", "118M"])
    parser.add_argument("--candidate_selection_method",type=str,   default="mass",
                        choices=["mass", "fingerprint"])
    parser.add_argument("--annotate_peaks",              action="store_true")
    parser.add_argument("--n_threads",                 type=int,   default=16)
    parser.add_argument("--chunk_size",                type=int,   default=4096)
    parser.add_argument("--seed",                      type=int,   default=42)
    parser.add_argument("--overwrite",                 action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.dataset_name == "massspecgym":
        download_massspecgym(overwrite=args.overwrite)
    else:
        download_spectraverse(overwrite=args.overwrite)

    process_smiles(
        dataset_name=args.dataset_name,
        overwrite=args.overwrite,
        n_threads=args.n_threads,
        chunk_size=args.chunk_size,
    )

    split(
        dataset_name=args.dataset_name,
        split_method=args.split_method,
        overwrite=args.overwrite,
    )

    prepare_candidates(
        dataset_name=args.dataset_name,
        n_candidates=args.n_candidates,
        kind=args.candidate_selection_method,
        sources=args.sources,
        overwrite=args.overwrite,
        seed=args.seed,
    )
    
    if args.annotate_peaks:
        annotate_peaks(
            dataset_name=args.dataset_name,
            n_threads=args.n_threads,
            chunksize=args.chunk_size,
            overwrite=args.overwrite,
        )