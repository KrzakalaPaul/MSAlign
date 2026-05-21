import argparse
from models import *
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Train MSAlign model")

    # Data
    parser.add_argument("--labelled_dataset_name", type=str, required=True)
    parser.add_argument("--candidate_map_name",    type=str, required=True)
    parser.add_argument("--split_method",          type=str, default="formula")

    # Logging
    parser.add_argument("--wandb_project",   type=str,   default="MSAlign")
    parser.add_argument("--wandb_run_name",  type=str,   default=None)
    parser.add_argument("--no_logger",        action="store_true")
    
    # Model
    parser.add_argument("--model",       type=str,   default="MSAlign")
    parser.add_argument("--config",      type=str,   default="default")
    
    # Misc
    parser.add_argument("--n_workers",        type=int,   default=8)
    parser.add_argument("--batch_size_test",  type=int,   default=16) # At test time, we load ALL candidates for each spectrum, so we need to reduce the batch size to fit in memory

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    
    if args.model.lower() == "msalign":
        config = yaml.safe_load(open(f"models/MSAlign/configs/{args.config}.yaml", "r"))
        train_and_eval_MSAlign(args, config)
    elif args.model.lower() == "embcos":
        config = yaml.safe_load(open(f"models/EmbCos/configs/{args.config}.yaml", "r"))
        train_and_eval_EmbCos(args, config)
    else:
        raise ValueError(f"Unknown model: {args.model}")