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
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL, StableDiffusionControlNetPipeline
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
pipe = None
last_checkpoint = None
last_lora_path = None
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


class NoWatermark:
    def apply_watermark(self, img):
        return img


def load_pipe(args):
    if 'eden' in args.ckpt:
        return load_pipe_v1(args)

    global pipe
    start_time = time.time()

    location = args.ckpt
    if os.path.isdir(os.path.join(CHECKPOINTS_PATH, args.ckpt)):
        location = os.path.join(CHECKPOINTS_PATH, args.ckpt)
        safetensor_files = [f for f in os.listdir(location) if f.endswith(".safetensors")]
        if len(safetensor_files) == 1:
            load_from_single_file = True
            location = os.path.join(location, safetensor_files[0])
        else:
            load_from_single_file = False

    print("#############################################")
    print(f"Loading new SD pipeline from {location}..")

    if args.controlnet_path is not None: # Load controlnet sdxl
        #from diffusers import StableDiffusionControlNetImg2ImgPipeline
        full_controlnet_path = os.path.join(CONTROLNET_PATH, args.controlnet_path)
        print(f"Loading SDXL controlnet-pipeline from {full_controlnet_path}")

        # check if any "*.safetensors" file is inside full_controlnet_path dir:
        use_safetensors = False
        for file in os.listdir(full_controlnet_path):
            if file.endswith(".safetensors"):
                use_safetensors = True
                break

        controlnet = ControlNetModel.from_pretrained(
            full_controlnet_path,
            torch_dtype=torch.float16,
            use_safetensors = use_safetensors
        )
        
        if load_from_single_file:
            print("Loading from single file...")
            pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                location, safety_checker=None,
                torch_dtype=torch.float16, use_safetensors=True)

            pipe = StableDiffusionXLControlNetPipeline(
                vae = pipe.vae,
                text_encoder = pipe.text_encoder,
                text_encoder_2 = pipe.text_encoder_2,
                tokenizer = pipe.tokenizer,
                tokenizer_2 = pipe.tokenizer_2,
                unet = pipe.unet,
                controlnet = controlnet,
                scheduler = pipe.scheduler,
            )

        else:
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                location,
                controlnet=controlnet,
                torch_dtype=torch.float16, use_safetensors=True, 
                #variant="fp16"
            )
            

    else: # Load normal sdxl base ckpt (no controlnet)
        print(f"Creating new StableDiffusionXLImg2ImgPipeline using {args.ckpt}")

        if load_from_single_file:
            pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                location,
                torch_dtype=torch.float16, use_safetensors=True)
        else:
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                location, 
                torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
            )

    pipe.safety_checker = None
    pipe = pipe.to(_device)
    #pipe.enable_model_cpu_offload()

    if args.compile_unet:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    # Disable watermarking (causes red dot artifacts for SDXL pipelines)
    pipe.watermark = NoWatermark()

    print(f"Created new pipe in {(time.time() - start_time):.2f} seconds")
    print_model_info(pipe)
    return pipe

def get_pipe(args, force_reload = False):
    global pipe
    global last_checkpoint
    global last_lora_path
    global last_controlnet_path

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

def prepare_prompt_for_lora(prompt, lora_path, interpolation=False, verbose=True):
    orig_prompt = prompt

    # Helper function to read JSON
    def read_json_from_path(path):
        with open(path, "r") as f:
            return json.load(f)

    # Check existence of "special_params.json"
    if not os.path.exists(os.path.join(lora_path, "special_params.json")):
        raise ValueError("This concept is from an old lora trainer that was deprecated. Please retrain your concept for better results!")

    token_map = read_json_from_path(os.path.join(lora_path, "special_params.json"))
    training_args = read_json_from_path(os.path.join(lora_path, "training_args.json"))
    
    lora_name = str(training_args["name"])
    lora_name_encapsulated = "<" + lora_name + ">"
    trigger_text = training_args["trigger_text"]
    mode = training_args["mode"]

    # Helper function for multiple replacements
    def replace_in_string(s, replacements):
        for target, replacement in replacements.items():
            s = s.replace(target, replacement)
        return s

    # Handle different modes
    print(f"lora mode: {mode}")
    if mode != "style":
        replacements = {
            "<concept>": trigger_text,
            lora_name_encapsulated: trigger_text,
            lora_name_encapsulated.lower(): trigger_text,
            lora_name: trigger_text,
            lora_name.lower(): trigger_text
        }
        prompt = replace_in_string(prompt, replacements)
        if not any(key in prompt for key in replacements.keys()):
            prompt = trigger_text + ", " + prompt
    else:
        style_replacements = {
            "in the style of <concept>": "in the style of TOK",
            f"in the style of {lora_name_encapsulated}": "in the style of TOK",
            f"in the style of {lora_name_encapsulated.lower()}": "in the style of TOK",
            f"in the style of {lora_name}": "in the style of TOK",
            f"in the style of {lora_name.lower()}": "in the style of TOK"
        }
        prompt = replace_in_string(prompt, style_replacements)
        if "in the style of" not in prompt:
            prompt = prompt + ", in the style of TOK"
        
    # Final cleanup
        prompt = replace_in_string(prompt, {"<concept>": "TOK", lora_name_encapsulated: "TOK"})

    if interpolation and mode != "style":
        prompt = "TOK, " + prompt

    # Replace tokens based on token map
    prompt = replace_in_string(prompt, token_map)

    # Fix common mistakes
    fix_replacements = {
        ",,": ",",
        "  ": " ",
        " .": ".",
        " ,": ","
    }
    prompt = replace_in_string(prompt, fix_replacements)

    if verbose:
        print('-------------------------')
        print("Adjusted prompt for LORA:")
        print(orig_prompt)
        print('-- to:')
        print(prompt)
        print('-------------------------')

    return prompt




def update_pipe_with_lora(pipe, args):
    global last_lora_path

    if (args.lora_path == last_lora_path) or (not args.lora_path):
        return pipe
    
    start_time = time.time()

    if "pytorch_lora_weights.bin" in os.listdir(args.lora_path): # trained with diffusers trainer
        pipe.load_lora_weights(args.lora_path)

    else: # trained with closeofismo trainer
        with open(os.path.join(args.lora_path, "training_args.json"), "r") as f:
            training_args = json.load(f)
            lora_rank = training_args["lora_rank"]
        
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
                rank=lora_rank,
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


###################################################


def load_pipe_v1(args):
    global pipe
    start_time = time.time()
    
    if os.path.isdir(os.path.join(CHECKPOINTS_PATH, args.ckpt)):
        location = os.path.join(CHECKPOINTS_PATH, args.ckpt)
    else:
        location = args.ckpt

    print("#############################################")
    print(f"Creating new StableDiffusionImg2ImgPipeline using {args.ckpt}")

    if args.controlnet_path is not None:
        full_controlnet_path = os.path.join(CONTROLNET_PATH, args.controlnet_path)
        print(f"Loading SD controlnet-pipeline from {full_controlnet_path}")

        controlnet = ControlNetModel.from_pretrained(
            full_controlnet_path,
            torch_dtype=torch.float16,
            use_safetensors = True,
        )     
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            location,
            controlnet=controlnet,
            torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    location, safety_checker=None, #local_files_only=_local_files_only,
                    torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

    pipe.safety_checker = None
    pipe = pipe.to(_device)
    print(f"Created new pipe in {(time.time() - start_time):.2f} seconds")
    print_model_info(pipe)
    return pipe