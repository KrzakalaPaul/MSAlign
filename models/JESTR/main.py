from .datamodule import JESTER_Datamodule
from .model import JESTR
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
import wandb

def pretrain_JESTR(args, config):
    
    model = JESTR(config)
    
    datamodule_pretrain = JESTER_Datamodule(
        labelled_dataset_name=args.labelled_dataset_name,
        split_method=args.split_method,
        fold=args.fold,
        batch_size_test=args.batch_size_test,
        n_workers=args.n_workers,
        bin_width=config['bin_width'],
        max_mz=config['max_mz'],
        batch_size=config['batch_size'],
        mode='pretrain'
    )

    
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
        max_steps=config['n_max_steps'],
        callbacks=callbacks,
        logger=None if args.no_logger else logger,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule_pretrain)
    trainer.test(model, datamodule=datamodule_pretrain, ckpt_path="best")
    
    wandb.finish()
    
    return model

def finetune_JESTR(args, config, model):
    
    datamodule_finetune = JESTER_Datamodule(
        labelled_dataset_name=args.labelled_dataset_name,
        split_method=args.split_method,
        fold=args.fold,
        batch_size_test=args.batch_size_test,
        n_workers=args.n_workers,
        k_candidates=config['k_candidates'],
        bin_width=config['bin_width'],
        max_mz=config['max_mz'],
        batch_size=config['batch_size'],
        mode='finetune'
    )

    callbacks = [
        ModelCheckpoint(
            monitor="R@1 (val)",
            mode="max",
            save_top_k=1,
            filename="best-finetune-{epoch:02d}-{R@1 (val):.3f}",
            verbose=True,
        ),
    ]

    logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_run_name + "_finetune",
        config={**config, **vars(args)},
    )

    trainer = Trainer(
        accelerator="gpu", 
        gradient_clip_val=5.0,
        max_steps=config['n_max_steps_finetune'],
        callbacks=callbacks,
        logger=None if args.no_logger else logger,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule_finetune)
    trainer.test(model, datamodule=datamodule_finetune, ckpt_path="best")
    
    wandb.finish()
    
def train_and_eval_JESTR(args, config):
    
    model = pretrain_JESTR(args, config)
    model.mode = 'finetune' 
    finetune_JESTR(args, config, model)
