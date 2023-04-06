import os
import sys
from pathlib import Path

SD_PATH = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
LORA_PATH = os.path.join(SD_PATH, 'lora')
LORA_DIFFUSION_PATH = os.path.join(LORA_PATH, 'lora_diffusion')
sys.path.append(LORA_PATH)
sys.path.append(LORA_DIFFUSION_PATH)

import time
import torch
from safetensors.torch import safe_open, save_file
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionDepth2ImgPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_eden import StableDiffusionEdenPipeline
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import load_pipeline_from_original_stable_diffusion_ckpt
from diffusers.models import AutoencoderKL

from diffusers import (
    LMSDiscreteScheduler, 
    EulerDiscreteScheduler, 
    DDIMScheduler, 
    DPMSolverMultistepScheduler, 
    KDPM2DiscreteScheduler, 
    PNDMScheduler
)

from eden_utils import *
from settings import _device
from lora_diffusion import *

global pipe
global last_checkpoint
global last_lora_path
pipe = None
last_checkpoint = None
last_lora_path = None

def set_sampler(sampler_name, pipe):
    schedulers = {
        "klms": LMSDiscreteScheduler.from_config(pipe.scheduler.config), 
        "euler": EulerDiscreteScheduler.from_config(pipe.scheduler.config),
        "dpm": DPMSolverMultistepScheduler.from_config(pipe.scheduler.config),
        "kdpm2": KDPM2DiscreteScheduler.from_config(pipe.scheduler.config),
        "pndm": PNDMScheduler.from_config(pipe.scheduler.config),
        "ddim": DDIMScheduler.from_config(pipe.scheduler.config),
    }
    if sampler_name not in schedulers:
        print(f"Sampler {sampler_name} not found. Available samplers: {list(schedulers.keys())}")
        print("Falling back to Euler sampler.")
        sampler_name = "euler"

    pipe.scheduler = schedulers[sampler_name]
    return pipe


def load_pipe(args):
    global pipe
    start_time = time.time()
    try:
        if args.mode == "depth2img":
            print("Creating new StableDiffusionDepth2ImgPipeline..")
            pipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth", safety_checker=None, torch_dtype=torch.float16 if args.half_precision else torch.float32)
        else:
            print(f"Creating new StableDiffusionEdenPipeline using {args.ckpt}")
            vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").half() # Use the (slightly better) updated vae model from stability
            pipe = StableDiffusionEdenPipeline.from_pretrained(args.ckpt, safety_checker=None, local_files_only = False, torch_dtype=torch.float16 if args.half_precision else torch.float32, vae=vae)
            #pipe = StableDiffusionPipeline.from_pretrained(args.ckpt, safety_checker=None, torch_dtype=torch.float16 if args.half_precision else torch.float32, vae=vae)
    
    except Exception as e:
        print(e)
        print("Failed to load from pretrained, trying to load from checkpoint")
        pipe = load_pipeline_from_original_stable_diffusion_ckpt(args.ckpt, image_size = 512)

    pipe.safety_checker = None
    print(f"Created new pipe in {(time.time() - start_time):.2f} seconds")
    pipe = pipe.to(_device)
    pipe.enable_xformers_memory_efficient_attention()
    print_model_info(pipe)
    return pipe


def get_pipe(args, force_reload = False):
    global pipe
    global last_checkpoint
    global last_lora_path
    # create a persistent, global pipe object:

    if args.ckpt != last_checkpoint:
        force_reload = True
        last_checkpoint = args.ckpt

    if not args.lora_path and last_lora_path:
        force_reload = True

    if (pipe is None) or force_reload:
        del pipe
        torch.cuda.empty_cache()

        if args.activate_tileable_textures:
            patch_conv(padding_mode='circular')

        pipe = load_pipe(args)

    # Potentially update the pipe:
    pipe = set_sampler(args.sampler, pipe)
    pipe = update_pipe_with_lora(pipe, args)
    
    if args.lora_path is not None:
        tune_lora_scale(pipe.unet, args.lora_scale)
        tune_lora_scale(pipe.text_encoder, args.lora_scale)

    last_lora_path = args.lora_path

    return pipe


def update_pipe_with_lora(pipe, args):
    global last_lora_path

    if args.lora_path == last_lora_path:
        return pipe

    if not args.lora_path:
        return pipe

    start_time = time.time()

    patch_pipe(
        pipe,
        args.lora_path,
        patch_text=True,
        patch_ti=True,
        patch_unet=True,
    )

    print(f" ---> Updated pipe in {(time.time() - start_time):.2f}s using lora from {args.lora_path} with scale = {args.lora_scale:.2f}")

    return pipe.to(_device)