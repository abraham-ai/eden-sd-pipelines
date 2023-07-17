import os
import sys
from pathlib import Path

SD_PATH = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
ROOT_PATH = SD_PATH.parents[0]
DIFFUSERS_PATH = os.path.join(ROOT_PATH, 'diffusers_sdxl/diffusers')
CHECKPOINTS_PATH = os.path.join(SD_PATH, 'models/checkpoints')
LORA_PATH = os.path.join(SD_PATH, 'lora')
LORA_DIFFUSION_PATH = os.path.join(LORA_PATH, 'lora_diffusion')

sys.path.insert(0,DIFFUSERS_PATH)
sys.path.append(LORA_PATH)
sys.path.append(LORA_DIFFUSION_PATH)

import time
import torch
from safetensors.torch import safe_open, save_file
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionDepth2ImgPipeline
#from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_eden import StableDiffusionEdenPipeline
from diffusers.models import AutoencoderKL
from diffusers.models.cross_attention import AttnProcessor2_0

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

global upscaling_pipe
global upscaling_last_checkpoint
global upscaling_last_lora_path
upscaling_pipe = None
upscaling_last_checkpoint = None
upscaling_last_lora_path = None

_local_files_only = False


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
    
    if os.path.isdir(os.path.join(CHECKPOINTS_PATH, args.ckpt)):
        location = os.path.join(CHECKPOINTS_PATH, args.ckpt)
    else:
        location = args.ckpt
        
    if args.mode == "depth2img":
        print(f"Creating new StableDiffusionDepth2ImgPipeline..")
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth", 
            safety_checker=None, 
            torch_dtype=torch.float16 if args.half_precision else torch.float32
        )
    else:
        print(f"Creating new DiffusionPipeline using {args.ckpt}")

        pipe = DiffusionPipeline.from_pretrained(
            location, 
            safety_checker=None, 
            #local_files_only=_local_files_only, 
            torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )

    pipe.safety_checker = None
    pipe = pipe.to(_device)
    pipe.unet.set_attn_processor(AttnProcessor2_0())
    print(f"Created new pipe in {(time.time() - start_time):.2f} seconds")
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


"""
same as the above, but specifically for img2img pipes (upscaler)
"""

def load_upscaling_pipe(args):
    global upscaling_pipe
    start_time = time.time()

    if os.path.isdir(os.path.join(CHECKPOINTS_PATH, args.ckpt)):
        load_path = os.path.join(CHECKPOINTS_PATH, args.ckpt)
    else:
        load_path = args.ckpt

    try:
        upscaling_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            load_path, 
            local_files_only = _local_files_only, 
            torch_dtype=torch.float16 if args.half_precision else torch.float32
        )
    except:
        upscaling_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            load_path, 
            local_files_only = _local_files_only, 
            torch_dtype=torch.float16 if args.half_precision else torch.float32
        )

    upscaling_pipe.unet.set_attn_processor(AttnProcessor2_0())
    upscaling_pipe = upscaling_pipe.to(_device)

    # Reduces max memory footprint:
    #upscaling_pipe.vae.enable_tiling()

    print(f"Created new img2img pipe from {load_path} in {(time.time() - start_time):.2f} seconds")
    return upscaling_pipe

def get_upscaling_pipe(args, force_reload = False):
    global upscaling_pipe
    global upscaling_last_checkpoint
    global upscaling_last_lora_path

    # create a persistent, global upscaling_pipe object:

    if args.ckpt != upscaling_last_checkpoint:
        force_reload = True
        upscaling_last_checkpoint = args.ckpt

    if not args.lora_path and upscaling_last_lora_path:
        force_reload = True

    if (upscaling_pipe is None) or force_reload:
        del upscaling_pipe
        torch.cuda.empty_cache()

        if args.activate_tileable_textures:
            patch_conv(padding_mode='circular')

        upscaling_pipe = load_upscaling_pipe(args)

    # Potentially update the pipe:
    upscaling_pipe = set_sampler(args.sampler, upscaling_pipe)
    upscaling_pipe = update_pipe_with_lora(upscaling_pipe, args)
    
    if args.lora_path is not None:
        tune_lora_scale(upscaling_pipe.unet, args.lora_scale)
        tune_lora_scale(upscaling_pipe.text_encoder, args.lora_scale)

    upscaling_last_lora_path = args.lora_path
    set_sampler("euler", upscaling_pipe)

    return upscaling_pipe
