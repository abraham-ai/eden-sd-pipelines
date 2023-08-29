import os
import sys
import shutil
from pathlib import Path

SD_PATH = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
ROOT_PATH = SD_PATH.parents[0]
print("SD_PATH: ", SD_PATH)
print("ROOT_PATH: ", ROOT_PATH)

DIFFUSERS_PATH = os.path.join(ROOT_PATH, 'diffusers')
CHECKPOINTS_PATH = os.path.join(SD_PATH, 'models/checkpoints')
CONTROLNET_PATH = os.path.join(SD_PATH, 'models/controlnets')
LORA_PATH = os.path.join(SD_PATH, 'lora')
LORA_DIFFUSION_PATH = os.path.join(LORA_PATH, 'lora_diffusion')

#print("DIFFUSERS PATH: ", DIFFUSERS_PATH)
#sys.path.insert(0,DIFFUSERS_PATH)
sys.path.append(LORA_PATH)
sys.path.append(LORA_DIFFUSION_PATH)

import time
import torch
from safetensors.torch import safe_open, save_file
import diffusers
print("Importing diffusers from:")
print(diffusers.__file__)
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionDepth2ImgPipeline

from diffusers import (
    DDIMScheduler, 
    DDPMScheduler,
    LMSDiscreteScheduler, 
    EulerDiscreteScheduler, 
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler, 
    KDPM2DiscreteScheduler, 
    KDPM2AncestralDiscreteScheduler,
    PNDMScheduler
)

from eden_utils import *
from settings import _device
from lora_diffusion import *

global pipe
global last_checkpoint
global last_lora_path
global last_controlnet_path
global last_lora_token_map
pipe = None
last_checkpoint = None
last_lora_path = None
last_lora_token_map = None
last_controlnet_path = None

global upscaling_pipe
global upscaling_last_checkpoint
global upscaling_last_lora_path
upscaling_pipe = None
upscaling_last_checkpoint = None
upscaling_last_lora_path = None

_local_files_only = False

# Disable gradients everywhere:
torch.set_grad_enabled(False)

def set_sampler(sampler_name, pipe):
    schedulers = {
        "ddim": DDIMScheduler.from_config(pipe.scheduler.config),
        "ddpm": DDPMScheduler.from_config(pipe.scheduler.config),
        "klms": LMSDiscreteScheduler.from_config(pipe.scheduler.config), 
        "euler": EulerDiscreteScheduler.from_config(pipe.scheduler.config),
        "euler_ancestral": EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config),
        "dpm": DPMSolverMultistepScheduler.from_config(pipe.scheduler.config),
        "kdpm2": KDPM2DiscreteScheduler.from_config(pipe.scheduler.config),
        "kdpm2_ancestral": KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config),
        "pndm": PNDMScheduler.from_config(pipe.scheduler.config),
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

    print("#############################################")
    print("Loading new SD pipeline..")

    if args.controlnet_path is not None: # Load controlnet sdxl
        #from diffusers import StableDiffusionControlNetImg2ImgPipeline

        print("Loading SDXL controlnet-pipeline..")

        controlnet = ControlNetModel.from_pretrained(
            os.path.join(CONTROLNET_PATH, args.controlnet_path),
            torch_dtype=torch.float16,
            use_safetensors = True if "depth" in args.controlnet_path else False,
        )

        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            location,
            controlnet=controlnet,
            torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )
        #pipe.enable_model_cpu_offload()

    else:
        print(f"Creating new StableDiffusionXLImg2ImgPipeline using {args.ckpt}")

        if args.ckpt == "dreamshaper": #dreamshaper
            location = "/data/xander/Projects/cog/eden-sd-pipelines/models/checkpoints/dreamshaper.safetensors"
            pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(location, safety_checker=None, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
            pipe = StableDiffusionXLImg2ImgPipeline(
                vae = pipe.vae,
                text_encoder = pipe.text_encoder,
                text_encoder_2 = pipe.text_encoder_2,
                tokenizer = pipe.tokenizer,
                tokenizer_2 = pipe.tokenizer_2,
                unet = pipe.unet,
                scheduler = pipe.scheduler)

        else: #SDXL 1.0
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                location, safety_checker=None, #local_files_only=_local_files_only,
                torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

    pipe.safety_checker = None
    pipe = pipe.to(_device)
    #pipe.enable_model_cpu_offload()

    if args.compile_unet:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    print(f"Created new pipe in {(time.time() - start_time):.2f} seconds")
    print_model_info(pipe)
    return pipe

def get_pipe(args, force_reload = False):
    global pipe
    global last_checkpoint
    global last_lora_path
    global last_controlnet_path
    # create a persistent, global pipe object:

    if args.ckpt != last_checkpoint:
        force_reload = True
        last_checkpoint = args.ckpt

    if args.controlnet_path != last_controlnet_path:
        force_reload = True
        last_controlnet_path = args.controlnet_path

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

    last_lora_path = args.lora_path

    return pipe

from safetensors import safe_open
from safetensors.torch import load_file
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from dataset_and_utils import TokenEmbeddingsHandler
def update_pipe_with_lora(pipe, args):
    global last_lora_path
    global last_lora_token_map

    if (args.lora_path == last_lora_path) or (not args.lora_path):
        return pipe
    
    start_time = time.time()

    if "pytorch_lora_weights.bin" in os.listdir(args.lora_path): # trained with diffusers trainer
        pipe.load_lora_weights(args.lora_path)

    else: # trained with closeofismo trainer
        print("Loading LORA token mapping:")
        with open(os.path.join(args.lora_path, "special_params.json"), "r") as f:
            token_map = json.load(f)
            print(json.dumps(token_map, indent=4, sort_keys=True))
            args.token_map = token_map

        unet = pipe.unet
        tensors = load_file(os.path.join(args.lora_path, "lora.safetensors"))
        unet_lora_attn_procs = {}

        for name, attn_processor in unet.attn_processors.items():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            module = LoRAAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=4,
            )
            unet_lora_attn_procs[name] = module.to("cuda")

        unet.set_attn_processor(unet_lora_attn_procs)
        unet.load_state_dict(tensors, strict=False)

        print("Adding new token embeddings to pipe...")
        handler = TokenEmbeddingsHandler(
                [pipe.text_encoder, pipe.text_encoder_2], [pipe.tokenizer, pipe.tokenizer_2]
            )
        handler.load_embeddings(os.path.join(args.lora_path, "embeddings.pti"))

    
    print(f" ---> Updated pipe in {(time.time() - start_time):.2f}s using lora from {args.lora_path} with scale = {args.lora_scale:.2f}")

    return pipe


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

    upscaling_pipe = upscaling_pipe.to(_device)
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
    #set_sampler("euler", upscaling_pipe)

    return upscaling_pipe
