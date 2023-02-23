import time
import torch
from safetensors.torch import safe_open, save_file
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionDepth2ImgPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_eden import StableDiffusionEdenPipeline
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import load_pipeline_from_original_stable_diffusion_ckpt
from diffusers.models import AutoencoderKL

from eden_utils import *
from settings import _device

pipe = None
last_checkpoint = None
last_lora_path = None


def load_pipe(args, img2img = False):
    global pipe
    start_time = time.time()
    try:
        if args.mode == "depth2img":
            print("Creating new StableDiffusionDepth2ImgPipeline..")
            pipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth", safety_checker=None, torch_dtype=torch.float16 if args.half_precision else torch.float32)
        else:
            print(f"Creating new StableDiffusionEdenPipeline using {args.ckpt}")
            vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").half() # Use the (slightly better) updated vae model from stability
            pipe = StableDiffusionEdenPipeline.from_pretrained(args.ckpt, safety_checker=None, local_files_only = True, torch_dtype=torch.float16 if args.half_precision else torch.float32, vae=vae)
            #pipe = StableDiffusionPipeline.from_pretrained(args.ckpt, safety_checker=None, torch_dtype=torch.float16 if args.half_precision else torch.float32, vae=vae)
    
    except Exception as e:
        print(e)
        print("Failed to load from pretrained, trying to load from checkpoint")
        pipe = load_pipeline_from_original_stable_diffusion_ckpt(args.ckpt, image_size = 512)

    pipe.safety_checker = None
    print(f"Created new pipe in {(time.time() - start_time):.2f} seconds")
    return pipe.to(_device)


def get_pipe(args, force_reload = False):
    # create a persistent, global pipe object:
    global pipe
    global last_checkpoint
    img2img = args.init_image is not None

    if args.ckpt != last_checkpoint:
        force_reload = True
        last_checkpoint = args.ckpt        

    if (pipe is None) or force_reload:
        del pipe
        torch.cuda.empty_cache()

        if args.activate_tileable_textures:
            patch_conv(padding_mode='circular')

        pipe = load_pipe(args, img2img = img2img)
        print_model_info(pipe)

    pipe = update_pipe_with_lora(pipe, args)
    pipe.enable_xformers_memory_efficient_attention()

    return pipe


def update_pipe_with_lora(pipe, args):
    global last_lora_path

    if args.lora_path == last_lora_path:
        return pipe

    start_time = time.time()
    patch_pipe(
        pipe,
        args.lora_path,
        patch_text=True,
        patch_ti=True,
        patch_unet=True,
    )
    tune_lora_scale(pipe.unet, args.lora_scale)
    tune_lora_scale(pipe.text_encoder, args.lora_scale)

    took_s = time.time() - start_time
    print(f" ---> Updated pipe in {took_s:.2f}s using lora from {args.lora_path} with scale = {args.lora_scale:.2f}")
    last_lora_path = args.lora_path

    safeloras = safe_open(args.lora_path, framework="pt", device="cpu")
    tok_dict = parse_safeloras_embeds(safeloras)

    trained_tokens = list(tok_dict.keys())

    return pipe.to(_device)