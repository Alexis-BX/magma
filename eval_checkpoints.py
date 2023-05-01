import torch
import os
import deepspeed
import wandb
from torch.utils.data import random_split, ConcatDataset
from torch.optim import AdamW
from tqdm import tqdm
from functools import partial
from magma.datasets import (
    collate_fn,
    ImgCptDataset,
)
from magma.magma import (
    Magma,
)
from magma.utils import (
    is_main,
    cycle,
    get_world_info,
    parse_args,
    wandb_log,
    wandb_init,
    save_model,
    load_model,
    print_main,
    configure_param_groups,
)
from magma.webdataset import get_wds_dataset
from magma.train_loop import (
    eval_step,
    inference_step,
    train_step,
)

import deepspeed.comm as dist
from deepspeed.runtime.utils import see_memory_usage

def get_pretraining_dataloader(config, tokenizer, transforms):
    
    def preprocess_text(text):
        return tokenizer.encode(
                    text,
                    return_tensors="pt",
                    max_length=2048,
                    padding="max_length",
                    truncation=True,)
    config.world_size=int(os.environ['WORLD_SIZE'])
    data = get_wds_dataset(config, transforms, preprocess_text, is_train=True)
    data.set_epoch(0) # [TODO]go change this when training more than 1 epoch
    train_loader=data.dataloader
    
    data = get_wds_dataset(config, transforms, preprocess_text, is_train=False)
    data.set_epoch(0)
    val_loader = data.dataloader

    return train_loader, val_loader

# tell tokenizers not to do parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":

    # parse command line arguments:
    args = parse_args()
    deepspeed.init_distributed()

    args.local_rank, _, _ = get_world_info()
    
    device=torch.device("cuda",args.local_rank)

    user = "alexisroger"
    experiment = "MAGMA_19M_clipH_5"
    
    path = f"/gpfs/alpine/csc499/proj-shared/magma/checkpoints/{user}/{experiment}"
    for checkpoint in path.iterdir():
        if checkpoint.suffix!='.pt':
            continue
        # load model + tokenizer:
        model = Magma.from_checkpoint(
            config_path = args.config,
            checkpoint_path = checkpoint,
            device=device
        )

        tokenizer, config, transforms = model.tokenizer, model.config, model.transforms

        # filter frozen from trainable parameters:
        trainable_parameters = configure_param_groups(model, config)

        opt = AdamW(
            trainable_parameters,
            config.lr,
            betas=(0.9, 0.95),
            weight_decay=config.weight_decay,
        )

        model_engine, opt, _, lr_scheduler = deepspeed.initialize(
            args=args,
            model=model,
            optimizer=opt,
            model_parameters=trainable_parameters,
            collate_fn=partial(collate_fn, seq_len=model.seq_len),
            config_params=config.deepspeed_config_params,
        )

        _, eval_loader = get_pretraining_dataloader(
        config, tokenizer, transforms
        )
        eval_loader = cycle(eval_loader)

        # initialize training
        global_step = 0
        if config.load:
            # loads a deepspeed checkpoint if provided. For finetuning, set load_optimizer to false
            previous_global_step = load_model(
                model_engine,
                config.load,
                load_optimizer_states=config.load_optimizer,
                load_lr_scheduler_states=config.load_optimizer,
            )

            if config.load_optimizer:
                global_step = previous_global_step

        pbar = tqdm(
            range(0, config.train_steps),
            desc="training...",
            initial=global_step,
            total=config.train_steps,
            disable=not is_main(),
        )
        wandb_init(
            project=config.wandb_project,
            name=config.name or wandb.util.generate_id(),
            config=config,
        )

        ##### Evaluation phase
        model_engine.eval()
        with torch.no_grad():
            ##### eval step:
            eval_loss = eval_step(config, eval_loader, model_engine)

            wandb_log({"eval/loss": eval_loss}, step=global_step)
            pbar.set_description(
                f"evaluating... Step: {global_step} Eval Loss: {eval_loss}"
            )
        
        print(checkpoint.stem, eval_loss)

