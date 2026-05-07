from preprocessing import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute spectra and molecule embeddings for MSAlign")

    parser.add_argument("--dataset_name",       type=str, required=True)
    parser.add_argument("--candidate_map_name", type=str, required=True)
    parser.add_argument("--version",            type=str, default="100M", choices=["13M", "100M"])
    parser.add_argument("--batch_size",         type=int, default=32)
    parser.add_argument("--chunk_size",         type=int, default=32)
    parser.add_argument("--n_workers",          type=int, default=4)
    parser.add_argument("--overwrite",          action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    get_dreams_embeddings(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        overwrite=args.overwrite,
    )

    get_chemberta_embeddings(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        version=args.version,
        overwrite=args.overwrite,
    )

    get_chemberta_embeddings_for_candidates(
        dataset_name=args.dataset_name,
        candidate_map_name=args.candidate_map_name,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        version=args.version,
        overwrite=args.overwrite,
    )