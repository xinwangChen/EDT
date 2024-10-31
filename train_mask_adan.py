"""
Masking training script for EDT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from adan import Adan
from accelerate import Accelerator
from torch.optim.lr_scheduler import LambdaLR
from models_edt import EDT_models
from diffusion import create_diffusion
from diffusion.resample import create_named_schedule_sampler, LossAwareSampler
from diffusers.models import AutoencoderKL
from load_model import find_model
import random

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)

def adjust_lr(opt, curr_step, total_step=400000, init_lr=5e-4, min_lr=5e-5):#4e-5
    if curr_step<=total_step:#
        lr = init_lr - (init_lr-min_lr)*curr_step/total_step
        for param_group in opt.param_groups:
            param_group['lr'] = lr


def forward_backward(model, batch, cond, microbatch, schedule_sampler, diffusion, accelerator, mini_step, opt):
    device = accelerator.device
    batch_total_loss = torch.tensor(0.0, device=device)
    batch_loss = torch.tensor(0.0, device=device)
    batch_mask_loss = torch.tensor(0.0, device=device)

    opt.zero_grad()
    for i in range(0, batch.shape[0], microbatch):
        micro = batch[i : i + microbatch]
        micro_batch_size = micro.shape[0]
        micro_cond = {k: v[i : i + microbatch] for k, v in cond.items()}
        t, weights = schedule_sampler.sample(micro.shape[0], device)

        losses = diffusion.training_losses(model, micro, t, model_kwargs=micro_cond)
        
        # mask stretagy of MDTv2
        # micro_cond_mask = micro_cond.copy()
        # micro_cond_mask['enable_input_mask'] = True
        # losses_mask = diffusion.training_losses(model, micro, t, model_kwargs=micro_cond_mask)

        micro_cond_mask = micro_cond.copy()
        micro_cond_mask['enable_down_mask'] = True
        losses_mask = diffusion.training_losses(model, micro, t, model_kwargs=micro_cond_mask)

        if isinstance(schedule_sampler, LossAwareSampler):
                schedule_sampler.update_with_local_losses(t, losses["loss"].detach()  + losses_mask["loss"].detach())


        losses_mask = (losses_mask["loss"] * weights).mean()
        losses = (losses["loss"] * weights).mean()
        total_loss = losses + losses_mask
        
        accelerator.backward(total_loss/mini_step)
        

        batch_total_loss = batch_total_loss + total_loss.detach()
        batch_loss = batch_loss + losses.detach()
        batch_mask_loss = batch_mask_loss + losses_mask.detach()
    
    opt.step()
    return batch_total_loss/mini_step, batch_loss/mini_step, batch_mask_loss/mini_step

# If out of memory, use this function.
# def forward_backward(model, batch, cond, microbatch, schedule_sampler, diffusion, accelerator, mini_step, opt):
#     device = accelerator.device
#     batch_total_loss = torch.tensor(0.0, device=device)
#     batch_loss = torch.tensor(0.0, device=device)
#     batch_mask_loss = torch.tensor(0.0, device=device)

#     opt.zero_grad()
#     for i in range(0, batch.shape[0], microbatch):
#         micro = batch[i : i + microbatch]
#         micro_batch_size = micro.shape[0]
#         micro_cond = {k: v[i : i + microbatch] for k, v in cond.items()}
#         t, weights = schedule_sampler.sample(micro.shape[0], device)

#         losses = diffusion.training_losses(model, micro, t, model_kwargs=micro_cond)
        
#         a_loss = (losses["loss"] * weights).mean()
#         accelerator.backward(a_loss/mini_step)

#         # mask stretagy of MDTv2
#         # micro_cond_mask = micro_cond.copy()
#         # micro_cond_mask['enable_input_mask'] = True
#         # losses_mask2 = diffusion.training_losses(model, micro, t, model_kwargs=micro_cond_mask)
        
#         micro_cond_mask = micro_cond.copy()
#         micro_cond_mask['enable_down_mask'] = True
#         losses_mask2 = diffusion.training_losses(model, micro, t, model_kwargs=micro_cond_mask)

#         b_loss = (losses_mask2["loss"] * weights).mean()
#         accelerator.backward(b_loss/mini_step)

#         if isinstance(schedule_sampler, LossAwareSampler):
#                 schedule_sampler.update_with_local_losses(t, losses_mask2["loss"].detach())

#         total_loss = a_loss + b_loss
        

#         batch_total_loss = batch_total_loss + total_loss.detach()
#         batch_loss = batch_loss + a_loss.detach()
#         batch_mask_loss = batch_mask_loss + b_loss.detach()
    
#     opt.step()
#     return batch_total_loss/mini_step, batch_loss/mini_step, batch_mask_loss/mini_step



#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new EDT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device
    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., EDT-XL/2 --> EDT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = EDT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    # if args.resume_step>0:
    #     state_dict = find_model(args.checkpoint_path)
    #     model.load_state_dict(state_dict)
    #     ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    #     state_dict = find_model(args.checkpoint_path,'model')
    #     model.load_state_dict(state_dict)
    #     model = model.to(device)
    # else:
    #     model = model.to(device)
    #     ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    
    model = model.to(device)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    vae = AutoencoderKL.from_pretrained(f"./sd-vae-ft-{args.vae}").to(device)
    if accelerator.is_main_process:
        logger.info(f"EDT Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    init_lr = args.init_lr #5e-4
    weight_decay=0.0
    # opt = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    opt = Adan(model.parameters(), lr=init_lr, weight_decay=weight_decay, max_grad_norm=0.0, fused=True)#Adan优化器
   
    if args.resume_step>0:
        state_dict = find_model(args.checkpoint_path,'opt')
        opt.load_state_dict(state_dict)
        adjust_lr(opt, args.resume_step, total_step=400000, init_lr=init_lr) #更新学习率
    
    # Setup data:
    features_dir = f"{args.feature_path}/imagenet_features"
    labels_dir = f"{args.feature_path}/imagenet_labels"
    dataset = CustomDataset(features_dir, labels_dir)
    batch_size=int(args.global_batch_size // accelerator.num_processes)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    microbatch = int((args.mini_batch_size if args.mini_batch_size > 0 else args.global_batch_size)//accelerator.num_processes)
    
    mini_step=1
    if args.mini_batch_size>0:
        if args.global_batch_size%args.mini_batch_size==0:
            mini_step=args.global_batch_size//args.mini_batch_size
        else:
            mini_step=args.global_batch_size//args.mini_batch_size + 1
    
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images ({args.feature_path})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    model, opt, loader = accelerator.prepare(model, opt, loader)
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_total_loss = torch.tensor(0.0, device=device)
    running_loss = torch.tensor(0.0, device=device)
    running_mask_loss = torch.tensor(0.0, device=device)
    start_time = time()
    train_steps=args.resume_step
        
    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):

        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.squeeze(dim=1)
            y = y.squeeze(dim=1)
            cond = dict(y=y)
            batch_total_loss, batch_loss, batch_mask_loss = forward_backward(model, x, cond, microbatch, schedule_sampler, diffusion, accelerator, mini_step, opt)
            # if train_steps<200000:
            #     clip_grad_norm_(accelerator.unwrap_model(model).parameters(), max_norm=2.0)
            update_ema(ema, model)
            log_steps += 1
            train_steps += 1
            running_total_loss += batch_total_loss
            running_loss += batch_loss
            running_mask_loss += batch_mask_loss
            adjust_lr(opt, train_steps, total_step=400000,init_lr=init_lr) #更新学习率

            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_total_loss = running_total_loss / log_steps
                avg_loss = running_loss / log_steps
                avg_mask_loss = running_mask_loss / log_steps

                avg_total_loss = accelerator.reduce(avg_total_loss, reduction="sum")
                avg_loss = accelerator.reduce(avg_loss, reduction="sum")
                avg_mask_loss = accelerator.reduce(avg_mask_loss, reduction="sum")

                if accelerator.is_main_process:
                    num_processes = accelerator.num_processes
                    avg_total_loss = avg_total_loss.item() / num_processes
                    avg_loss = avg_loss.item() / num_processes
                    avg_mask_loss = avg_mask_loss.item() / num_processes
                    logger.info(f"(step={train_steps:07d}) Avg Total Loss: {avg_total_loss:.4f}, Avg Loss: {avg_loss:.4f}, Avg Mask Loss: {avg_mask_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_total_loss = torch.tensor(0.0, device=device)
                running_loss = torch.tensor(0.0, device=device)
                running_mask_loss = torch.tensor(0.0, device=device)
                log_steps = 0
                start_time = time()

            # Save EDT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
            # if train_steps % 1000000 == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train EDT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="features256")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--checkpoint-path", type=str, default='')
    parser.add_argument("--schedule_sampler", type=str, choices=["uniform", "loss-second-moment"], default="loss-second-moment")
    parser.add_argument("--model", type=str, choices=list(EDT_models.keys()), default="EDT-S/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--mini-batch-size", type=int, default=-1)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--init-lr", type=float, default=1e-3)
    parser.add_argument("--resume-step", type=int, default=0)
    args = parser.parse_args()
    main(args)
