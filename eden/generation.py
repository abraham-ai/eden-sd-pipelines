import os
import sys
from pathlib import Path

SD_PATH = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
ROOT_PATH = SD_PATH.parents[0]
CLIP_INTERROGATOR_MODEL_PATH = os.path.join(ROOT_PATH, 'cache')
LORA_PATH = os.path.join(ROOT_PATH, 'lora')
sys.path.append(LORA_PATH)

from _thread import start_new_thread
from queue import Queue
from copy import copy
import time
import numpy as np
import random
from PIL import Image
from einops import rearrange
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from settings import StableDiffusionSettings, _device
from eden_utils import *
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionDepth2ImgPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_eden import StableDiffusionEdenPipeline
from diffusers import LMSDiscreteScheduler, EulerDiscreteScheduler, DDIMScheduler, DPMSolverMultistepScheduler, KDPM2DiscreteScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import load_pipeline_from_original_stable_diffusion_ckpt
from diffusers.models import AutoencoderKL

from interpolator import *
from clip_tools import *
from planner import LatentTracker, create_init_latent, blend_inits

from lora_diffusion import tune_lora_scale, patch_pipe, monkeypatch_or_replace_safeloras, monkeypatch_remove_lora, dict_to_lora, load_safeloras_both, apply_learned_embed_in_clip, parse_safeloras, monkeypatch_or_replace_lora_extended, parse_safeloras_embeds

# some global variables that persist between function calls:
pipe = None
last_checkpoint = None
last_lora_path = None

from safetensors.torch import safe_open, save_file

"""
class LoraBlender():
    # Helper class to blend LORA models on the fly during interpolations

    def __init__(self, lora_scale):
        self.lora_scale = lora_scale
        self.loras_in_memory = {}
        self.embeds_in_memory = {}

    def load_lora(self, lora_path):
        if lora_path in self.loras_in_memory:
            return self.loras_in_memory[lora_path], self.embeds_in_memory[lora_path]
        else:
            print(f" ---> Loading lora from {lora_path} into memory..")
            safeloras = safe_open(lora_path, framework="pt", device=device)
            embeddings = parse_safeloras_embeds(safeloras)
            
            self.loras_in_memory[lora_path] = safeloras
            self.embeds_in_memory[lora_path] = embeddings

            return safeloras, embeddings

    def blend_embeds(self, embeds_1, embeds_2, t):
        # Blend the two dictionaries of embeddings:
        ret_embeds = {}
        for key in set(list(embeds_1.keys()) + list(embeds_2.keys())):
            if key in embeds_1.keys() and key in embeds_2.keys():
                ret_embeds[key] = (1-t) * embeds_1[key] + t * embeds_2[key]
            elif key in embeds_1.keys():
                ret_embeds[key] = embeds_1[key]
            elif key in embeds_2.keys():
                ret_embeds[key] = embeds_2[key]
        return ret_embeds

    def patch_pipe(self, pipe, t, lora1_path, lora2_path):
        print(f" ---> Patching pipe with lora1 = {os.path.basename(os.path.dirname(lora1_path))} and lora2 = {os.path.basename(os.path.dirname(lora2_path))} at t = {t:.2f}")

        # Load the two loras:
        safeloras_1, embeds_1 = self.load_lora(lora1_path)
        safeloras_2, embeds_2 = self.load_lora(lora2_path)

        metadata = dict(safeloras_1.metadata())
        metadata.update(dict(safeloras_2.metadata()))
        
        # Combine / Linear blend the token embeddings:
        blended_embeds = self.blend_embeds(embeds_1, embeds_2, t)

        # Blend the two loras:
        ret_tensor = {}
        for keys in set(list(safeloras_1.keys()) + list(safeloras_2.keys())):
            if keys.startswith("text_encoder") or keys.startswith("unet"):
                tens1 = safeloras_1.get_tensor(keys)
                tens2 = safeloras_2.get_tensor(keys)
                ret_tensor[keys] = (1-t) * tens1 + t * tens2
            else:
                if keys in safeloras_1.keys():
                    tens = safeloras_1.get_tensor(keys)
                else:
                    tens = safeloras_2.get_tensor(keys)
                ret_tensor[keys] = tens

        loras = dict_to_lora(ret_tensor, metadata)

        # Apply this blended lora to the pipe:
        for name, (lora, ranks, target) in loras.items():
            model = getattr(pipe, name, None)
            if not model:
                print(f"No model provided for {name}, contained in Lora")
                continue
            print("Patching model", name, "with LORA")
            monkeypatch_or_replace_lora_extended(model, lora, target, ranks)

        apply_learned_embed_in_clip(
            blended_embeds,
            pipe.text_encoder,
            pipe.tokenizer,
            token=None,
            idempotent=True,
        )

        # Set the lora scale:
        tune_lora_scale(pipe.unet, self.lora_scale)
        tune_lora_scale(pipe.text_encoder, self.lora_scale)

        return blended_embeds
"""

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



def load_pipe(args, img2img = False):
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
    #print(f"Sampler set to {sampler_name}")


def maybe_apply_watermark(args, x_images):
    # optionally, apply watermark to final image:
    if args.watermark_path is not None:
        # check if args.watermarker already exists:
        if not hasattr(args, 'watermarker'):
            # get width and height of image:
            pil_img = x_images[0]
            W, H = pil_img.size
            args.watermarker = WaterMarker(W, H, args.watermark_path) 
        # apply watermark:
        x_images = args.watermarker.apply_watermark(x_images)
    return x_images


@torch.no_grad()
def generate(
    args, 
    upscale = False,
):
    assert args.text_input is not None

    seed_everything(args.seed)
    args.W = round_to_nearest_multiple(args.W, 64)
    args.H = round_to_nearest_multiple(args.H, 64)

    # Load init image
    if args.init_image_data:
        args.init_image = load_img(args.init_image_data, 'RGB')

    if args.init_image is not None:
        #args.W, args.H = match_aspect_ratio(args.W * args.H, args.init_image)
        args.init_image = args.init_image.resize((args.W, args.H), Image.LANCZOS)

    force_starting_latent = None
    if args.interpolator is not None:
        # Create a new trajectory for the latent tracker:
        args.interpolator.latent_tracker.create_new_denoising_trajectory(args)
        force_starting_latent = args.interpolator.latent_tracker.force_starting_latent
    
    # Load model
    pipe = get_pipe(args)
    set_sampler(args.sampler, pipe)
    
    # if init image strength == 1, just return the initial image
    if args.init_image_strength == 1.0 and args.init_image:
        latent = pil_img_to_latent(args.init_image, args, _device, pipe)
        if args.interpolator is not None:
            args.interpolator.latent_tracker.add_latent(latent, init_image_strength = 1.0)

        pt_images = T.ToTensor()(args.init_image).unsqueeze(0).to(_device)
        pil_images = [args.init_image] * args.n_samples
        
        if args.upscale_f != 1.0:
            pt_images, pil_images = run_upscaler(args, pil_images)

        pil_images = maybe_apply_watermark(args, pil_images)
        return pt_images, pil_images

    # get the denoising schedule:
    n_steps = max(args.steps, int(args.min_steps/(1-args.init_image_strength)))
    n_steps = args.steps

    # Callback
    callback_ = make_callback(
        latent_tracker = args.interpolator.latent_tracker if args.interpolator is not None else None,
        extra_callback = None,
    )

    generator = torch.Generator(device=_device).manual_seed(args.seed)
    #generator = None

    if args.c is not None:
        prompt, negative_prompt = None, None
        #seed_everything(0)
    else:
        prompt, negative_prompt = args.text_input, args.uc_text
        args.c, args.uc = None, None

    if 0:
        for token in ["<person1>", "<person2>"]:
            print("Patching token", token)
            token_id = pipe.tokenizer.convert_tokens_to_ids(token)
            embed = pipe.text_encoder.get_input_embeddings().weight.data[token_id]
            print(embed[-510:-500])


    if args.mode == 'depth2img':
        pipe_output = pipe(
            prompt = prompt, 
            image = args.init_image,
            strength = 1-args.init_image_strength,
            #depth_map = None,
            negative_prompt = negative_prompt,
            num_inference_steps = n_steps,
            guidance_scale = args.guidance_scale,
            num_images_per_prompt = args.n_samples,
            )
    else:
        if args.init_image is not None or True:   # img2img / Eden
            pipe_output = pipe(
                prompt = prompt,
                negative_prompt = negative_prompt, 
                width = args.W, 
                height = args.H,
                image=args.init_image, 
                strength=1-args.init_image_strength, 
                num_inference_steps = n_steps,
                guidance_scale = args.guidance_scale,
                num_images_per_prompt = args.n_samples,
                prompt_embeds = args.c,
                negative_prompt_embeds = args.uc,
                generator = generator,
                latents = args.init_latent,
                force_starting_latent = force_starting_latent,
                callback = callback_,
            )
        else:   # text2img
            pipe_output = pipe(
                prompt = prompt, 
                negative_prompt = negative_prompt,
                width = args.W, 
                height = args.H,
                num_inference_steps = n_steps,
                guidance_scale = args.guidance_scale,
                num_images_per_prompt = args.n_samples,
                latents = args.init_latent,
                prompt_embeds = args.c,
                negative_prompt_embeds = args.uc,
                generator = generator,
            )
        
    pil_images = pipe_output.images

    try:
        final_latents = pipe_output.final_latents
    except:
        final_latents = None
        
    pt_images = [None]*len(pil_images)

    if args.interpolator is not None and final_latents is not None:  # add the final denoised latent to the tracker:
        args.interpolator.latent_tracker.add_latent(final_latents, init_image_strength = args.init_image_strength)

    if args.upscale_f != 1.0:
        print(f"Upscaling with f = {args.upscale_f:.3f}...")
        pt_images, pil_images = run_upscaler(args, pil_images)

    pil_images = maybe_apply_watermark(args, pil_images)
    return pt_images, pil_images


@torch.no_grad()
def make_interpolation(args, force_timepoints = None):
    
    if args.text_input == "real2real" and args.interpolation_texts:
        assert len(args.interpolation_texts) == len(args.interpolation_init_images), "When overwriting prompts for real2real, you must provide the same number of interpolation texts as init_imgs!"
        real2real_texts = args.interpolation_texts
    else:
        real2real_texts = None

    if not args.interpolation_texts:
        args.interpolation_texts = [args.text_input]
        if args.interpolation_init_images:
            args.interpolation_texts = args.interpolation_texts * len(args.interpolation_init_images)
    if not args.interpolation_init_images:
        args.interpolation_init_images = [None]
        if args.interpolation_texts:
            args.interpolation_init_images = args.interpolation_init_images * len(args.interpolation_texts)
    if not args.interpolation_seeds:
        args.interpolation_seeds = [args.seed]
        args.n_frames = 1

    assert args.n_samples==1, "Batch size >1 not implemented yet"
    assert len(args.interpolation_texts) == len(args.interpolation_seeds)
    assert len(args.interpolation_init_images) == len(args.interpolation_seeds)

    if args.loop and len(args.interpolation_texts) > 2:
        args.interpolation_texts.append(args.interpolation_texts[0])
        args.interpolation_seeds.append(args.interpolation_seeds[0])
        args.interpolation_init_images.append(args.interpolation_init_images[0])

    global pipe
    pipe = get_pipe(args)
    #model = update_aesthetic_gradient_settings(model, args)

    #loraBlender = LoraBlender(args.lora_scale) if args.lora_paths is not None else None
    
    # if there are init images, change width/height to their average
    interpolation_init_images = None
    if args.interpolation_init_images and all(args.interpolation_init_images):
        assert len(args.interpolation_init_images) == len(args.interpolation_texts), "Number of initial images must match number of prompts"
        
        args.use_init = True
        interpolation_init_images = get_uniformly_sized_crops(args.interpolation_init_images, args.H * args.W)
        args.W, args.H = interpolation_init_images[0].size

        # if args.interpolation_init_images_use_img2txt, then use clip-interrogator to overwrite interpolation_texts
        if args.interpolation_init_images_use_img2txt:
            if real2real_texts is None:
                init_img_prompts = [clip_interrogate(args.ckpt, init_img, args.clip_interrogator_mode, CLIP_INTERROGATOR_MODEL_PATH) for init_img in interpolation_init_images]
                args.interpolation_texts = init_img_prompts
                print("Overwriting prompts with clip-interrogator results:", init_img_prompts)
            else:
                args.interpolation_texts = real2real_texts

            # We're in Real2Real mode here --> overwrite args.aesthetic_target with the interpolation_init_images
            # This activates aesthetic gradient finetuning of the individual prompt conditioning vectors on each single init_image:
            #args.aesthetic_target = [[img] for img in interpolation_init_images]

    else:
        args.use_init = False

    del_clip_interrogator_models()

    print("Creating interpolator")
    print(args.n_frames, "frames")

    args.interpolator = Interpolator(
        pipe, 
        args.interpolation_texts, 
        args.n_frames, 
        args, 
        _device, 
        smooth=args.smooth,
        seeds=args.interpolation_seeds,
        scales=[args.guidance_scale for _ in args.interpolation_texts],
        scale_modulation_amplitude_multiplier=args.scale_modulation,
    )

    #args.n_anchor_imgs = max(3, int(args.anchor_img_fraction * args.interpolator.n_frames_between_two_prompts))
    #args.n_anchor_imgs = min(args.n_anchor_imgs, 7)
    print("Using", args.n_anchor_imgs, "anchor images per prompt pair")

    n_frames = len(args.interpolator.ts)
    if force_timepoints is not None:
        n_frames = len(force_timepoints)

    active_lora_path = args.lora_paths[0] if args.lora_paths is not None else None

    ######################################

    for f in range(n_frames):
        if force_timepoints is not None:
            force_t_raw = force_timepoints[f]
        else:
            force_t_raw = None

        if False: # catch errors and try to complete the video
            try:
                t, t_raw, c, init_latent, scale, return_index, _, _ = args.interpolator.get_next_conditioning(verbose=0, save_distances_to_dir = args.save_distances_to_dir, t_raw = force_t_raw)
            except Exception as e:
                print("Error in interpolator.get_next_conditioning(): ", str(e))
                break
        else: # get full stack_trace, for debugging:
            t, t_raw, c, init_latent, scale, return_index, _, _ = args.interpolator.get_next_conditioning(verbose=0, save_distances_to_dir = args.save_distances_to_dir, t_raw = force_t_raw)
        
        args.c = c
        args.guidance_scale = scale
        args.t_raw = t_raw

        if args.lora_paths is not None:
            # Maybe update the lora to the next person:
            if args.lora_paths[return_index] != active_lora_path:
                active_lora_path = args.lora_paths[return_index]
                print("Switching to lora path", active_lora_path)
                args.lora_path = active_lora_path

        #if loraBlender is not None:
        #    lora_path1, lora_path2 = args.lora_paths[return_index], args.lora_paths[(return_index + 1) % len(args.lora_paths)]
        #    loraBlender.patch_pipe(pipe, t, lora_path1, lora_path2)

        if 1 and (args.interpolation_init_images and all(args.interpolation_init_images) or len(args.interpolator.latent_tracker.frame_buffer.ts) >= args.n_anchor_imgs):

            if interpolation_init_images is None: # lerping mode (no init imgs)
                is_real2real = False
                init_img1, init_img2 = args.interpolator.latent_tracker.frame_buffer.get_current_keyframe_imgs()
                init_img1, init_img2 = sample_to_pil(init_img1), sample_to_pil(init_img2)
            else: # real2real mode
                is_real2real = True
                init_img1, init_img2 = interpolation_init_images[return_index], interpolation_init_images[return_index + 1]
            
            if len(args.interpolator.latent_tracker.frame_buffer.ts) < args.n_anchor_imgs and is_real2real and 0:
                print("Pixel blending...")
                # apply linear blending of keyframe images in pixel space and then encode
                args.init_image, args.init_image_strength = blend_inits(init_img1, init_img2, t, args, real2real = is_real2real)
                args.init_latent = None
            else:
                # perform Latent-Blending initialization:
                args.init_latent, args.init_image, args.init_image_strength = create_init_latent(args, t, init_img1, init_img2, _device, pipe, real2real = is_real2real)

        else: #only use the raw init_latent noise from interpolator (using the input seeds)
            print("Using raw init noise (strenght 0.0)...")
            args.init_latent = init_latent
            args.init_image = None
            args.init_image_strength = 0.0

        if args.planner is not None: # When audio modulation is active:
            args = args.planner.adjust_args(args, t_raw, force_timepoints=force_timepoints)

        print(f"Interpolating frame {f+1}/{len(args.interpolator.ts)} (t_raw = {t_raw:.5f},\
                init_strength: {args.init_image_strength:.2f},\
                latent skip_f: {args.interpolator.latent_tracker.latent_blending_skip_f:.2f},\
                splitting lpips_d: {args.interpolator.latent_tracker.frame_buffer.get_perceptual_distance_at_t(args.t_raw):.2f}),\
                keyframe {return_index+1}/{len(args.interpolation_texts) - 1}...")


        args.lora_path = active_lora_path
        _, pil_images = generate(args)
        img_pil = pil_images[0]
        img_t = T.ToTensor()(img_pil).unsqueeze_(0).to(_device)
        args.interpolator.latent_tracker.add_frame(args, img_t, t, t_raw)

        yield img_pil, t_raw

    # Flush the final metadata to disk if needed:
    args.interpolator.latent_tracker.reset_buffer()


def make_images(args):

    if args.mode == "remix":
        enable_random_lr_flipping = True  # randomly flip the init img for remixing?

        if args.init_image_data:
            args.init_image = load_img(args.init_image_data, 'RGB')

        assert args.init_image is not None, "Must provide an init image in order to remix it!"
        
        if random.random() > 0.5 and enable_random_lr_flipping:
            args.init_image = args.init_image.transpose(Image.FLIP_LEFT_RIGHT)

        args.W, args.H = match_aspect_ratio(args.W * args.H, args.init_image)
        args.aesthetic_target = [args.init_image]
        args.text_input = clip_interrogate(args.ckpt, args.init_image, args.clip_interrogator_mode, CLIP_INTERROGATOR_MODEL_PATH)

        del_clip_interrogator_models()

    assert args.text_input is not None

    #global pipe            
    #pipe = get_pipe(args)
    #pipe = update_aesthetic_gradient_settings(pipe, args)
    
    _, images_pil = generate(args)
    
    return images_pil


def make_callback(
    latent_tracker=None,
    extra_callback=None,
):
    # Callback for _call_ in diffusers repo, called thus:
    #   callback(i, t, latents)
    def diffusers_callback(i, t, latents):
        if latent_tracker is not None:
            latent_tracker.add_latent(latents)
        if extra_callback and False: # TODO fix this function for discord etc...
            x = model.decode_first_stage(args_dict['x'])
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255. * rearrange(x, 'b c h w -> b h w c')
            x = x.cpu().numpy().astype(np.uint8)
            x = [Image.fromarray(x_) for x_ in x]
            extra_callback(x, args_dict['i'])
              
    return diffusers_callback


def run_upscaler(args_, imgs, init_image_strength = 0.7, min_steps = 30):
    args = copy(args_)
    args.n_samples = 1  # batching will prob cause OOM, so run everything in a loop
    args.init_image_data = None
    args.init_latent = None
    if args.interpolator is not None:
        args.interpolator.latent_tracker = None
    args.init_image_strength = init_image_strength
    args.steps = int(min_steps/(1-args.init_image_strength))
    args.W, args.H = args_.upscale_f * args_.W, args_.upscale_f * args_.H
    args.upscale_f = 1.0  # don't upscale again

    x_samples_upscaled, x_images_upscaled = [], []

    for i in range(len(imgs)): # upscale in a loop:
        args.init_image = imgs[i]
        x_samples, x_images = generate(args)
        x_samples_upscaled.extend(x_samples)
        x_images_upscaled.extend(x_images)

    return x_samples_upscaled, x_images_upscaled


def interrogate(args):
    if args.init_image_data:
        args.init_image = load_img(args.init_image_data, 'RGB')
    
    assert args.init_image is not None, "Must provide an init image"
    interrogated_prompt = clip_interrogate(args.ckpt, args.init_image, args.clip_interrogator_mode, CLIP_INTERROGATOR_MODEL_PATH)
    #del_clip_interrogator_models()

    return interrogated_prompt
    