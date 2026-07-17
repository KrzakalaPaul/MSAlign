import argparse
import math
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from FLARE import FLARE_model, FLARE_Finetune_Datamodule
from JESTR import JESTR_model, JESTR_Finetune_Datamodule
import argparse
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

torch.set_printoptions(precision=2, sci_mode=False) 

def parse_args():
    parser = argparse.ArgumentParser(description="Train MS-CLIP baseline")
    
    # ---------------- Model ---------------- # 
    
    parser.add_argument("--model", type=str, default="FLARE", help="Model Name")
    
    # ---------------- Optimization ---------------- # 
    
    parser.add_argument("--base_lr", type=float, default=1e-5, help="Base learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for optimizer")
    parser.add_argument("--n_warmup_steps", type=int, default=1000, help="Number of warmup steps for scheduler")
    parser.add_argument("--n_max_steps", type=int, default=10000, help="Total number of training steps")
    parser.add_argument("--no_scheduler", action='store_true', help="Whether to use learning rate scheduler")
    
    # ---------------- Training Objective ---------------- #
    
    # Batch composition
    parser.add_argument("--batchsize", type=int, default=32, help="Training batch size")
    parser.add_argument("--batchsize_all_candidates", type=int, default=8, help="Batch size when all candidates are used")
    parser.add_argument("--n_max_candidates", type=int, default=32)
        
    # ---------------- Hardware dependent choices ---------------- #
    
    parser.add_argument("--n_workers", type=int, default=16, help="Number of data loading workers")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Dataloader prefetch factor")

    # ---------------- Logging ---------------- #
    parser.add_argument("--run_name", type=str, default="finetune", help="WandB or internal run name")
    parser.add_argument("--n_eval_checks", type=int, default=8, help="Number of evaluation checks to perform during training")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--n_test_batches", type=int, default=100, help="Number of test batches to run during training at the end of training")
    
    # ---------------- Checkpoint ---------------- #
    parser.add_argument("--checkpoint_path", type=str, default='checkpoints/pretrain/best.ckpt', help="Path to model checkpoint to initialize from (for fine-tuning or resuming training)")

    return parser.parse_args()

def load_data(args, model):
    params = model.hparams
    args.labelled_dataset = params['labelled_dataset']
    args.split_method = params['split_method']
    args.candidate_dataset_train = params['candidate_dataset_train']
    args.candidate_dataset_eval = params['candidate_dataset_eval']
    
    if args.model == "FLARE":
        datamodule_class = FLARE_Finetune_Datamodule
    elif args.model == "JESTR":
        datamodule_class = JESTR_Finetune_Datamodule
    
    datamodule = datamodule_class(
        labelled_dataset_name=params['labelled_dataset'],
        split_method=params['split_method'],
        candidate_dataset_train_name=params['candidate_dataset_train'],
        candidate_dataset_eval_name=params['candidate_dataset_eval'],
        batch_size=args.batchsize, 
        batchsize_all_candidates=args.batchsize_all_candidates,
        n_workers=args.n_workers, 
        prefetch_factor=args.prefetch_factor,
        n_max_candidates=args.n_max_candidates
    )
    return datamodule

def load_model(args):
    if args.model == "FLARE":
        model = FLARE_model.load_from_checkpoint(args.checkpoint_path, weights_only=False)
    elif args.model == "JESTR":
        model = JESTR_model.load_from_checkpoint(args.checkpoint_path, weights_only=False)
    model.mode = 'finetune'
    return model

def main():
    args = parse_args()
    args.mode = 'finetune'
    model = load_model(args)
    datamodule = load_data(args, model)
   
    
    print(f"Data train size: {len(datamodule.train_dataset)}")
    print(f"Data valid size: {len(datamodule.val_dataset)}")
    print(f"Data test size: {len(datamodule.test_dataset)}")
    
    tags = [f'labelled:{args.labelled_dataset}', f'candidates:{args.candidate_dataset_eval}', f'split:{args.split_method}',f'version:4.0', 'JESTR']

    wandb_logger = WandbLogger(
        project="MS-CLIP",
        name=args.run_name,
        save_dir="logs/wandb_logs",
        tags=tags
    )
    if args.debug:
        wandb_logger = None
    
    checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{args.run_name}", 
                                          save_top_k=1, 
                                          monitor="val_acc (finetune)", 
                                          mode="max", 
                                          filename="best")
    
    val_check_interval = args.n_max_steps // args.n_eval_checks
    if val_check_interval > len(datamodule.train_dataloader()):
        check_val_every_n_epoch = math.ceil(val_check_interval / len(datamodule.train_dataloader()))
        val_check_interval = None
    else:
        check_val_every_n_epoch = None

    trainer = pl.Trainer(max_steps=args.n_max_steps,
                         accelerator="gpu", 
                         logger=wandb_logger,
                         check_val_every_n_epoch=check_val_every_n_epoch,
                         val_check_interval=val_check_interval,
                         gradient_clip_val=1.0, 
                         callbacks=[checkpoint_callback],
                         limit_val_batches=args.n_test_batches,
                         limit_test_batches=args.n_test_batches
                         ) 
    
    trainer.test(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule) 
    best_model_path = checkpoint_callback.best_model_path
    if args.model == "FLARE":
        model = FLARE_model.load_from_checkpoint(best_model_path, weights_only=False)
    elif args.model == "JESTR":
        model = JESTR_model.load_from_checkpoint(best_model_path, weights_only=False)
    trainer.test(model, datamodule=datamodule)
        
if __name__ == "__main__":
    main()