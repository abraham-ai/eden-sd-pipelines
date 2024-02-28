import os
import sys
import shutil
from pathlib import Path
import gc
import re

SD_PATH = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
ROOT_PATH = SD_PATH.parents[0]
print("SD_PATH: ", SD_PATH)
print("ROOT_PATH: ", ROOT_PATH)

DIFFUSERS_PATH = os.path.join(ROOT_PATH, 'diffusers')
CHECKPOINTS_PATH = os.path.join(SD_PATH, 'models/checkpoints')
CONTROLNET_PATH = os.path.join(SD_PATH, 'models/controlnets')
LORA_PATH  = os.path.join(SD_PATH, 'models/loras')
IP_ADAPTER_PATH = os.path.join(SD_PATH, 'models/ip_adapter/ip-adapter_sdxl.bin')
IP_ADAPTER_IMG_ENCODER_PATH = os.path.join(SD_PATH, 'models/ip_adapter/image_encoder')

import time
import torch
from safetensors.torch import safe_open, save_file
import diffusers
print("Importing diffusers from:")
print(diffusers.__file__)
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL, StableDiffusionXLControlNetImg2ImgPipeline
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionDepth2ImgPipeline

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPConfig

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

from ip_adapter.ip_adapter import IPAdapterXL
from eden_utils import *
from settings import _device

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


_download_dict = {
    "models/controlnets/controlnet-luminance-sdxl-1.0": "https://edenartlab-lfs.s3.amazonaws.com/models/controlnets/controlnet-luminance-sdxl-1.0/diffusion_pytorch_model.bin",
    "models/controlnets/controlnet-depth-sdxl-1.0-small": "https://edenartlab-lfs.s3.amazonaws.com/models/controlnets/controlnet-depth-sdxl-1.0-small/diffusion_pytorch_model.safetensors",
    "models/checkpoints/sdxl-v1.0": "https://edenartlab-lfs.s3.amazonaws.com/models/checkpoints/sdxl-v1.0/sd_xl_base_1.0_0.9vae.safetensors"
}

from io_utils import download
def maybe_download(path):
    print("maybe_download: ", path)
    for key in _download_dict:
        if key in path:
            print(f"{key} in path, might download...")
            download_url = _download_dict[key]
            filename = os.path.basename(download_url)
            
            # check if path is a folder or a file:
            if os.path.isdir(path):
                folderpath = path
            else:
                folderpath = os.path.dirname(path)

            local_path = os.path.join(folderpath, filename)
            print(f"target local_path: {local_path}")
            download(download_url, "", filepath=local_path, timeout=20*60)
            return
    if not os.path.exists(path):
        print("Warning, no download option found for path: ", path)

class NoWatermark:
    def apply_watermark(self, img):
        return img

class PipeManager:
    # utility class to manage a consistent pipe object
    def __init__(self):
        self.pipe = None
        self.safety_checker = None
        self.last_checkpoint = None
        self.last_lora_path = None
        self.last_controlnet_path = None
        self.ip_adapter = None

    def get_pipe(self, args, force_reload = False):
        if (args.ckpt != self.last_checkpoint) or (args.controlnet_path != self.last_controlnet_path):
            force_reload = True

        if not args.lora_path and self.last_lora_path: # for now, unsetting lora requires reloading...
            force_reload = True

        if (self.pipe is None) or force_reload:
            self.clear()

            if args.activate_tileable_textures:
                patch_conv(padding_mode='circular')

            self.pipe, self.safety_checker = load_pipe(args)
            self.last_checkpoint = args.ckpt
            self.last_lora_path = None # load_pipe does not set lora
            self.last_controlnet_path = args.controlnet_path
            self.ip_adapter = None

            if args.use_lcm:
                from diffusers import LCMScheduler
                self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
                self.pipe.to(_device)
                adapter_id = os.path.join(LORA_PATH, "lcm_sdxl_lora.safetensors")
                #adapter_id = "latent-consistency/lcm-lora-ssd-1b"
                print(f"Loading lcm-loa from {adapter_id}...")
                self.pipe.load_lora_weights(adapter_id)
                self.pipe.fuse_lora()

        self.update_pipe_with_lora(args)

        if not args.use_lcm:
            self.pipe = set_sampler(args.sampler, self.pipe)

        return self.pipe

    def run_safety_checker(self, image): #deprecated
        if self.safety_checker is not None:
            safety_checker_input = self.pipe.feature_extractor(self.pipe.numpy_to_pil(image), return_tensors="pt").to(_device)
            image, nsfw_detected, watermark_detected = self.safety_checker(
                images=image,
                clip_input=safety_checker_input.pixel_values, #.to(dtype=dtype),
            )
        return nsfw_detected

    def check_is_nsfw(self, image): #deprecated
        if self.safety_checker is not None:
            safety_checker_input = self.pipe.feature_extractor(self.pipe.numpy_to_pil(image), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        else:
            has_nsfw_concept = None

        if has_nsfw_concept:
            print("Detected nsfw concept!!")

        return has_nsfw_concept

    def update_pipe_with_lora(self, args):
        if (args.lora_path != self.last_lora_path) and args.lora_path:
            self.pipe = load_lora(self.pipe, args)
            self.last_lora_path = args.lora_path

    def enable_ip_adapter(self, force_reload = False):
        if self.ip_adapter and not force_reload:
            self.ip_adapter.enable_ip_adapter()
        else:
            self.ip_adapter = IPAdapterXL(self.pipe, IP_ADAPTER_IMG_ENCODER_PATH, IP_ADAPTER_PATH, _device)

        return self.ip_adapter

    def disable_ip_adapter(self):
        if self.ip_adapter:
            self.ip_adapter.disbable_ip_adapter()

    def clear(self):
        del self.pipe
        del self.last_checkpoint
        del self.last_lora_path
        del self.last_controlnet_path
        del self.ip_adapter
        self.pipe = None
        self.last_checkpoint = None
        self.last_lora_path = None
        self.last_controlnet_path = None
        self.ip_adapter = None
        gc.collect()
        torch.cuda.empty_cache()

pipe_manager = PipeManager()

def load_pipe(args):
    use_dtype = torch.float16

    if 'eden-v1' in os.path.basename(args.ckpt):
        return load_pipe_v1(args), None

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
    else:
        # load from hf hub:
        load_from_single_file = False

    maybe_download(location)

    print("#############################################")
    print(f"Loading new SD pipeline from {location}..")
    print("#############################################")

    if args.controlnet_path is not None: # Load controlnet sdxl
        #from diffusers import StableDiffusionControlNetImg2ImgPipeline
        full_controlnet_path = os.path.join(CONTROLNET_PATH, args.controlnet_path)
        print(f"Loading SDXL controlnet-pipeline from {full_controlnet_path}")

        maybe_download(full_controlnet_path)
        controlnet = ControlNetModel.from_pretrained(
            full_controlnet_path,
            torch_dtype=use_dtype,
            use_safetensors = any(file.endswith(".safetensors") for file in os.listdir(full_controlnet_path))
        )
        
        if load_from_single_file:
            print("Loading from single file...")
            pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                location, torch_dtype=use_dtype, use_safetensors=True)

            pipe = StableDiffusionXLControlNetImg2ImgPipeline(
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
            pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                location,
                controlnet=controlnet,
                torch_dtype=torch.float16, use_safetensors=True, 
            )
            
    else: # Load normal sdxl base ckpt (no controlnet)
        if load_from_single_file:
            print(f"Loading SDXL from single file: {location}...")
            pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                location,
                torch_dtype=use_dtype, use_safetensors=True)
        else:
            #location = "segmind/SSD-1B"
            print(f"Loading SDXL from pretrained: {location}...")
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                location, 
                torch_dtype=use_dtype, use_safetensors=True, variant="fp16"
            )

    safety_checker = None
    pipe.safety_checker = None

    pipe = pipe.to(_device)
    #pipe.enable_model_cpu_offload()

    # Disable watermarking (causes red dot artifacts for SDXL pipelines)
    pipe.watermark = NoWatermark()

    print(f"Created new pipe in {(time.time() - start_time):.2f} seconds")
    print_model_info(pipe)
    return pipe, safety_checker

from safetensors import safe_open
from safetensors.torch import load_file
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from dataset_and_utils import TokenEmbeddingsHandler

import re

def read_json_from_path(path):
    """Helper function to read JSON from a given path."""
    with open(path, "r") as f:
        return json.load(f)

def fix_common_mistakes(prompt):
    """Fix common mistakes in the prompt."""
    replacements = {
        r",,": ",",
        r"\s\s+": " ",
        r"\s\.": ".",
        r"\s,": ","
    }
    return replace_in_string(prompt, replacements)

def replace_in_string(s, replacements):
    while True:
        replaced = False
        for target, replacement in replacements.items():
            new_s = re.sub(target, replacement, s, flags=re.IGNORECASE)
            if new_s != s:
                s = new_s
                replaced = True
        if not replaced:
            break
    return s

def blend_conditions(embeds1, embeds2, args, 
        token_scale_power = 0.5,  # adjusts the curve of the interpolation
        min_token_scale   = 0.5,  # minimum token scale (corresponds to lora_scale = 0)
        #min_token_scale   = 1.0,  # minimum token scale (corresponds to lora_scale = 0)
        verbose = False,
        ):

    if min_token_scale == 1.0:
        print("WARNING: min_token_scale = 1.0, this means that the lora token will never be fully disabled!")
        print("WARNING: min_token_scale = 1.0, this means that the lora token will never be fully disabled!")
        
    """
    using args.lora_scale (or args.token_scale), apply linear interpolation between two sets of embeddings
    """
    c1, uc1, pc1, puc1 = embeds1
    c2, uc2, pc2, puc2 = embeds2

    if not args.token_scale:
        args.token_scale = args.lora_scale ** token_scale_power
        # rescale the [0,1] range to [min_token_scale, 1] range:
        args.token_scale = min_token_scale + (1 - min_token_scale) * args.token_scale
        if verbose:
            print(f"Setting token_scale to {args.token_scale:.2f} (lora_scale = {args.lora_scale}, power = {token_scale_power})")
            print('-------------------------')

    c   = (1 - args.token_scale) * c1   + args.token_scale * c2
    uc  = (1 - args.token_scale) * uc1  + args.token_scale * uc2
    pc  = (1 - args.token_scale) * pc1  + args.token_scale * pc2
    puc = (1 - args.token_scale) * puc1 + args.token_scale * puc2

    return c, uc, pc, puc


def adjust_prompt(args, prompt, 
        inject_token = True,
        verbose = False,
        token = "TOK",
        ):
    """
    Slightly messy prompt magic to make sure we're always triggering the lora token when a lora is active
    """
    original_prompt = prompt

    training_args = read_json_from_path(os.path.join(args.lora_path, "training_args.json"))
    token_map     = read_json_from_path(os.path.join(args.lora_path, "special_params.json"))
    mode          = training_args.get("concept_mode", training_args.get("mode", "object"))
    lora_name     = training_args.get("name", "lora")
    lora_name_encapsulated = "<" + lora_name + ">"

    trigger_text = training_args.get("trigger_text", token)
    if len(trigger_text) == 0:
        trigger_text = token

    if inject_token:
        face_txt   = trigger_text
        object_txt = trigger_text
        style_txt  = f"in the style of {token}"
    else:
        face_txt   = "a person"
        object_txt = training_args.get("segmentation_prompt", "a thing")
        style_txt  = ""

    if mode == "face":
        replacements = {
            "<concept>": face_txt,
            "<concepts>": face_txt + "'s",
            lora_name_encapsulated: face_txt,
            lora_name: face_txt,
        }
        prompt = replace_in_string(prompt, replacements)
        if face_txt not in prompt:
            prompt = face_txt + ", " + prompt

    elif mode == "object" or mode == "concept":
        replacements = {
            "<concept>": object_txt,
            "<concepts>": object_txt + "'s",
            lora_name_encapsulated: object_txt,
            lora_name: object_txt,
        }
        prompt = replace_in_string(prompt, replacements)
        if object_txt not in prompt:
            prompt = object_txt + ", " + prompt

    elif mode == "style":
        style_replacements = {
            "in the style of <concept>": style_txt,
            f"in the style of {lora_name_encapsulated}": style_txt,
            f"in the style of {lora_name}": style_txt,
        }
        prompt = replace_in_string(prompt, style_replacements)

        style_replacements = {
            "<concept>": style_txt,
            f"{lora_name_encapsulated}": style_txt,
            f"{lora_name}": style_txt,
        }
        prompt = replace_in_string(prompt, style_replacements)

        if token not in prompt and inject_token:
            prompt = f"{style_txt}, " + prompt
        
    # Final cleanup
    if inject_token:
        prompt = replace_in_string(prompt, {"<concept>": token, lora_name_encapsulated: token, lora_name: token})
        if token not in prompt:
            prompt = f"{token}, " + prompt
            
        prompt = replace_in_string(prompt, token_map)
    prompt = fix_common_mistakes(prompt)

    if verbose:
        print('-------------------------')
        print(original_prompt)
        print(" -- adjusted to: --")
        print(prompt)

    return prompt

def encode_prompt_advanced(args, prompt_to_encode, pipe, ignore_set = False, verbose=True):
    if (args.c is not None) and (not ignore_set):
        print("Warning: args.c was already set, skipping prepare_prompt_for_lora()")
        return args.c, args.uc, args.pc, args.puc

    if args.lora_path is None or not os.path.exists(os.path.join(args.lora_path, "special_params.json")):
        return pipe.encode_prompt(
            prompt_to_encode,
            do_classifier_free_guidance=args.guidance_scale > 1,
            negative_prompt=args.uc_text)
    else:
        lora_prompt     = adjust_prompt(args, prompt_to_encode, verbose=verbose)
        args.text_input = lora_prompt

        zero_prompt        = adjust_prompt(args, prompt_to_encode, inject_token=False, verbose=verbose)
        zero_embeddings    = pipe.encode_prompt(
            zero_prompt,
            do_classifier_free_guidance=args.guidance_scale > 1,
            negative_prompt=args.uc_text,
            lora_scale=args.lora_scale)

        lora_embeddings    = pipe.encode_prompt(
            lora_prompt,
            do_classifier_free_guidance=args.guidance_scale > 1,
            negative_prompt=args.uc_text,
            lora_scale=args.lora_scale)

        embeds = blend_conditions(zero_embeddings, lora_embeddings, args, verbose=verbose)
        return embeds

def load_lora(pipe, args):
    
    start_time = time.time()

    if args.lora_path.endswith(".safetensors"):
        print("Loading lora from single file...")
        pipe.load_lora_weights(args.lora_path)
        args.lora_path += "_no_token"

    elif "pytorch_lora_weights.bin" in os.listdir(args.lora_path): # trained with diffusers trainer
        pipe.load_lora_weights(args.lora_path)

    else: # trained with Eden trainer
        with open(os.path.join(args.lora_path, "training_args.json"), "r") as f:
            training_args = json.load(f)
            lora_rank = training_args["lora_rank"]
        
        unet = pipe.unet

        lora_filename = [f for f in os.listdir(args.lora_path) if f.endswith("lora.safetensors")][0]
        tensors = load_file(os.path.join(args.lora_path, lora_filename))

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

        embedding_path = os.path.join(args.lora_path, "embeddings.pti")
        if not os.path.exists(embedding_path):
            embeddings_filename = [f for f in os.listdir(args.lora_path) if f.endswith("embeddings.safetensors")][0]
            embedding_path = os.path.join(args.lora_path, embeddings_filename)

        handler.load_embeddings(embedding_path)
    
    print(f" ---> Updated pipe in {(time.time() - start_time):.2f}s using lora from {args.lora_path} with scale = {args.lora_scale:.2f}")

    return pipe


###################################################


def load_pipe_v1(args):
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
                    location, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe = pipe.to(_device)

    print(f"Created new pipe in {(time.time() - start_time):.2f} seconds")
    print_model_info(pipe)
    return pipe