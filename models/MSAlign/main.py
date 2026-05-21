from .datamodule import MSCLIP_Datamodule
from .model import MSAlign
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer

def train_and_eval_MSAlign(args, config):

    datamodule = MSCLIP_Datamodule(
        labelled_dataset_name=args.labelled_dataset_name,
        candidate_map_name=args.candidate_map_name,
        split_method=args.split_method,
        batch_size_test=args.batch_size_test,
        n_workers=args.n_workers,
        k_candidates=config['k_candidates'],
        encoder_mol=config['encoder_mol'],
        encoder_spectra=config['encoder_spectra'],
        batch_size=config['batch_size'],
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

    trainer = Trainer(
        accelerator="gpu", 
        gradient_clip_val=5.0,
        max_steps=args.n_max_steps,
        callbacks=callbacks,
        logger=None if args.no_logger else logger,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")
