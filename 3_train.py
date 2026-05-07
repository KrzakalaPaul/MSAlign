import argparse
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from MSAlign.model import MSAlign
from MSAlign.datamodule import MSCLIP_Datamodule


def parse_args():
    parser = argparse.ArgumentParser(description="Train MSAlign model")

    # Data
    parser.add_argument("--labelled_dataset_name", type=str, required=True)
    parser.add_argument("--candidate_map_name",    type=str, required=True)
    parser.add_argument("--split_method",          type=str, default="formula")
    parser.add_argument("--k_candidates",          type=int, default=16)

    # Encoders
    parser.add_argument("--encoder_mol",     type=str, default="chemberta_13M",
                        choices=["chemberta_13M", "chemberta_100M", "morgan_2048_2", "grover", "MHG-GED"])
    parser.add_argument("--encoder_spectra", type=str, default="dreams")

    # Model architecture
    parser.add_argument("--d_hidden",        type=int,   default=2048)
    parser.add_argument("--d_shared",        type=int,   default=1024)
    parser.add_argument("--n_hidden_layers", type=int,   default=1)
    parser.add_argument("--dropout",         type=float, default=0.2)
    parser.add_argument("--layernorm",       action="store_true")

    # Training
    parser.add_argument("--batch_size",      type=int,   default=128)
    parser.add_argument("--batch_size_test", type=int,   default=16)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--weight_decay",    type=float, default=0.0)
    parser.add_argument("--n_max_steps",      type=int,   default=16000)
    parser.add_argument("--n_warmup_steps",      type=int,   default=2000)
    parser.add_argument("--n_workers",       type=int,   default=8)

    # Logging
    parser.add_argument("--wandb_project",   type=str,   default="MSAlign")
    parser.add_argument("--wandb_run_name",  type=str,   default=None)
    parser.add_argument("no_logger",        action="store_true")

    return parser.parse_args()


def resolve_embedding_dims(args):
    if args.encoder_mol == "morgan_2048_2":
        args.d_mol = 2048
    elif args.encoder_mol == "grover":
        args.d_mol = 2400
    elif args.encoder_mol == "MHG-GED":
        args.d_mol = 1024
    else:
        args.d_mol = 768
    args.d_ms = 1024
    return args


def build_config(args):
    return {
        "d_ms":            args.d_ms,
        "d_mol":           args.d_mol,
        "d_hidden":        args.d_hidden,
        "d_shared":        args.d_shared,
        "n_hidden_layers": args.n_hidden_layers,
        "dropout":         args.dropout,
        "layernorm":       args.layernorm,
        "lr":              args.lr,
        "weight_decay":    args.weight_decay,
        "n_warmup_steps":    args.n_warmup_steps,
        "n_max_steps":      args.n_max_steps,
    }


def main():
    args = parse_args()
    args = resolve_embedding_dims(args)
    config = build_config(args)

    datamodule = MSCLIP_Datamodule(
        labelled_dataset_name=args.labelled_dataset_name,
        candidate_map_name=args.candidate_map_name,
        split_method=args.split_method,
        k_candidates=args.k_candidates,
        encoder_mol=args.encoder_mol,
        encoder_spectra=args.encoder_spectra,
        batch_size=args.batch_size,
        batch_size_test=args.batch_size_test,
        n_workers=args.n_workers,
    )

    model = MSAlign(config)

    callbacks = [
        ModelCheckpoint(
            monitor="R@1 (val)",
            mode="max",
            save_top_k=1,
            filename="best-{epoch:02d}-{R@1 (val):.3f}",
            verbose=True,
        ),
    ]

    logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={**config, **vars(args)},
    )

    trainer = pl.Trainer(
        accelerator="gpu", 
        gradient_clip_val=5.0,
        max_steps=args.n_max_steps,
        callbacks=callbacks,
        logger=None if args.no_logger else logger,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()