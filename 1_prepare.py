from preprocessing import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for MSAlign training")

    parser.add_argument("--dataset_name",              type=str,   default="massspecgym",
                        choices=["massspecgym", "spectraverse"])
    parser.add_argument("--split_method",              type=str,   default="formula",
                        choices=["formula", "random", "as_provided", "inchi", "murcko", "murcko_scaffold", "murcko_hist"])
    parser.add_argument("--n_candidates",              type=int,   default=256)
    parser.add_argument("--candidate_selection_method",type=str,   default="mass",
                        choices=["mass", "formula"])
    parser.add_argument("--annotate_peaks",              action="store_true")
    parser.add_argument("--n_threads",                 type=int,   default=16)
    parser.add_argument("--chunk_size",                type=int,   default=4096)
    parser.add_argument("--seed",                      type=int,   default=42)
    parser.add_argument("--overwrite",                 action="store_true")
    parser.add_argument("--prepare_custom_candidates",       action="store_true", help="Whether to prepare a custom candidate map. If False MassSpecGym will download the official candidate map instead. For spectraverse, this flag is ignored and custom candidates are always prepared.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.dataset_name == "massspecgym":
        download_massspecgym(overwrite=args.overwrite)
        if not args.prepare_custom_candidates:
            download_massspecgym_official_candidate_map(overwrite=args.overwrite, n_threads=args.n_threads, chunk_size=args.chunk_size, kind=args.candidate_selection_method)
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
        seed=args.seed,
        add_seed_name=False,
    )

    if args.prepare_custom_candidates or args.dataset_name == "spectraverse":
        prepare_candidates(
            dataset_name=args.dataset_name,
            n_candidates=args.n_candidates,
            kind=args.candidate_selection_method,
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
