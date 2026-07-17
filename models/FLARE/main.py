from .datamodule import FLARE_Datamodule
from .model import FLARE
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer

def train_and_eval_FLARE(args, config):

    datamodule = FLARE_Datamodule(
        labelled_dataset_name=args.labelled_dataset_name,
        candidate_map_name=args.candidate_map_name,
        split_method=args.split_method,
        batch_size_test=args.batch_size_test,
        n_workers=args.n_workers,
        batch_size=config['batch_size'],
    )

    model = FLARE(config)

    callbacks = [
        ModelCheckpoint(
            monitor="R@1 - batch (val)",
            mode="max",
            save_top_k=1,
            filename="best-{epoch:02d}-{R@1 - batch (val):.3f}",
            verbose=True,
        ),
    ]

    logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={**config, **vars(args)},
    )

    trainer = Trainer(
        accelerator="gpu", 
        gradient_clip_val=5.0,
        max_steps=config['n_max_steps'],
        callbacks=callbacks,
        logger=None if args.no_logger else logger,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")
