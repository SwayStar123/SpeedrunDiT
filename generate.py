# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from models.sit import SiT_models
from preprocessing.encoders import load_invae
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from samplers import euler_maruyama_sampler, euler_maruyama_sampler_path_drop
from utils import download_model


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load model:
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    latent_size = args.resolution // 16  # invae uses 16x downsampling
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        in_channels=32,
        use_cfg=True,
        z_dims=[int(z_dim) for z_dim in args.projector_embed_dims.split(',')],
        **block_kwargs,
    ).to(device)
    # Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
    ckpt_path = args.ckpt


    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if ckpt_path is None:
        args.ckpt = download_model(repo_id="SwayStar123/SpeedrunDiT", filename="256/0400000.pt")
        assert args.model == 'SiT-B/1'
        assert len(args.projector_embed_dims.split(',')) == 1
        assert int(args.projector_embed_dims.split(',')[0]) == 768
        assert args.qk_norm is True
        assert args.resolution == 256
        assert args.mode == "sde"
        ckpt = torch.load(args.ckpt, map_location=f'cuda:{device}', weights_only=False)
        state_dict = ckpt['ema'] if isinstance(ckpt, dict) and 'ema' in ckpt else ckpt
    else:
        ckpt = torch.load(ckpt_path, map_location=f'cuda:{device}', weights_only=False)
        state_dict = ckpt['ema'] if isinstance(ckpt, dict) and 'ema' in ckpt else ckpt

    model.load_state_dict(state_dict)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    model.eval()  # important!
    # Load invae model using load_invae function
    if rank == 0:
        _ = load_invae("REPA-E/e2e-invae", device=torch.device("cpu"))
    dist.barrier()
    vae = load_invae("REPA-E/e2e-invae", device=torch.device(f"cuda:{device}"))


    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.resolution}-vae-invae-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}-{args.mode}-{args.guidance_high}-{args.cls_cfg_scale}-pathdrop-{args.path_drop}"
    if args.balanced_sampling:
        folder_name += "-balanced"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")

        # Check for existing PNG samples to optionally skip resampling.
        existing_pngs = [f for f in os.listdir(sample_folder_dir) if f.endswith(".png")]
        existing_count = len(existing_pngs)
        if existing_count >= args.num_fid_samples:
            print(
                f"Found {existing_count} existing PNG samples in {sample_folder_dir}, "
                f"skipping sampling and only rebuilding the .npz file."
            )
            need_sampling = False
        else:
            need_sampling = True
    else:
        need_sampling = True

    # Broadcast need_sampling decision from rank 0 to all ranks.
    need_sampling_tensor = torch.tensor(int(need_sampling), device=device)
    dist.broadcast(need_sampling_tensor, src=0)
    need_sampling = bool(need_sampling_tensor.item())
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0 and need_sampling:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"projector Parameters: {sum(p.numel() for p in model.projectors.parameters()):,}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)

    if need_sampling:
        pbar = range(iterations)
        pbar = tqdm(pbar) if rank == 0 else pbar
        total = 0
        for _ in pbar:
            # Sample inputs:
            z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
            if args.balanced_sampling:
                # Use global sample indices to assign labels evenly across classes.
                # This ensures each class index appears approximately equally often.
                indices = (torch.arange(n, device=device) * dist.get_world_size() + rank + total)
                y = (indices % args.num_classes).long()
            else:
                y = torch.randint(0, args.num_classes, (n,), device=device)
            cls_z = torch.randn(n, args.cls, device=device)

            # Sample images:
            sampling_kwargs = dict(
                model=model, 
                latents=z,
                y=y,
                num_steps=args.num_steps, 
                heun=args.heun,
                cfg_scale=args.cfg_scale,
                guidance_low=args.guidance_low,
                guidance_high=args.guidance_high,
                path_type=args.path_type,
                cls_latents=cls_z,
                args=args
            )
            with torch.no_grad():
                if args.mode == "sde":
                    if args.path_drop:
                        samples = euler_maruyama_sampler_path_drop(**sampling_kwargs).to(torch.float32)
                    else:
                        samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
                elif args.mode == "ode":# will support
                    exit()
                    #samples = euler_sampler(**sampling_kwargs).to(torch.float32)
                else:
                    raise NotImplementedError()

                # For invae, apply 0.3099 scaling factor
                scaling_factor = 0.3099
                samples = vae.decode(samples / scaling_factor).sample
                samples = (samples + 1) / 2.
                samples = torch.clamp(
                    255. * samples, 0, 255
                    ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

                # Save samples to disk as individual .png files
                for i, sample in enumerate(samples):
                    index = i * dist.get_world_size() + rank + total
                    Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
            total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=0)

    # precision
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    # logging/saving:
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a SiT checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="samples")

    # model
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-B/1")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=True)

    # number of samples
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)

    parser.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction, default=True,
                        help="If enabled, sample class labels in a balanced way so each class index appears equally often.")

    # sampling related hyperparameters
    parser.add_argument("--mode", type=str, default="sde")
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--cls-cfg-scale",  type=float, default=1.5)
    parser.add_argument("--projector-embed-dims", type=str, default="768")
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False) # only for ode
    parser.add_argument("--guidance-low", type=float, default=0.)
    parser.add_argument("--guidance-high", type=float, default=1.)
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--cls', default=768, type=int)
    parser.add_argument('--path-drop', default=True, action=argparse.BooleanOptionalAction,)

    parser.add_argument("--time-shifting", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shift-base", type=int, default=4096)


    args = parser.parse_args()
    main(args)
