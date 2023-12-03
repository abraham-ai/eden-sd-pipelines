import os
import glob
import sys
from pathlib import Path

SD_PATH = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
ROOT_PATH = SD_PATH.parents[0]
CHECKPOINTS_PATH = os.path.join(SD_PATH, 'models/checkpoints')
CLIP_INTERROGATOR_MODEL_PATH = os.path.join(ROOT_PATH, 'cache')
LORA_PATH = os.path.join(ROOT_PATH, 'lora')
DEPTH_PATH = os.path.join(SD_PATH, 'eden/depth')

sys.path.append(LORA_PATH)
sys.path.append(DEPTH_PATH)

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

from settings import _device
from pipe import pipe_manager, prepare_prompt_for_lora
from eden_utils import *
from interpolator import *
from clip_tools import *
from planner import LatentTracker, create_init_latent, blend_inits
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionDepth2ImgPipeline

def maybe_apply_watermark(args, x_images):
    # optionally, apply watermark to final image:
    if args.watermark_path:
        # check if args.watermarker already exists:
        if not hasattr(args, 'watermarker'):
            # get width and height of image:
            pil_img = x_images[0]
            W, H = pil_img.size
            args.watermarker = WaterMarker(W, H, args.watermark_path) 
        # apply watermark:
        x_images = args.watermarker.apply_watermark(x_images)
    return x_images

def make_callback(
    latent_tracker=None,
    extra_callback=None,
):
    def diffusers_callback(i, t, latents, pre_timestep = 0):
        if latent_tracker:
            latent_tracker.add_latent(i, t, latents, pre_timestep = pre_timestep)
              
    return diffusers_callback

@torch.no_grad()
def generate(
    args, 
    upscale = False,
    do_callback = False,
):
    seed_everything(args.seed)
    args.init_image_strength = float(args.init_image_strength)
    free_memory, tot_mem = torch.cuda.mem_get_info(device=_device)
    print(f"Start of generate, free memory: {free_memory / 1e9:.2f} GB")

    if args.init_image == "":
        args.init_image = None

    if args.n_target_pixels is None:
        args.n_target_pixels = args.W * args.H

    # Load init images
    if args.init_image is not None:
        args.init_image_path = args.init_image
        args.init_image = load_img(args.init_image)

    if args.control_image is not None:
        args.control_image_path = args.control_image
        args.control_image = load_img(args.control_image)

    if args.adopt_aspect_from_init_img:
        if args.init_image is not None:
            args.W, args.H = match_aspect_ratio(args.n_target_pixels, args.init_image)
        elif args.control_image is not None:
            args.W, args.H = match_aspect_ratio(args.n_target_pixels, args.control_image)
    
    args.W = round_to_nearest_multiple(args.W, 8)
    args.H = round_to_nearest_multiple(args.H, 8)

    if args.init_image is not None:
        args.init_image = args.init_image.resize((args.W, args.H), Image.LANCZOS)
    if args.control_image is not None:
        args.control_image = args.control_image.resize((args.W, args.H), Image.LANCZOS)

    # Load model
    pipe = pipe_manager.get_pipe(args)

    if args.ip_image and not args.lora_path:
        print(f"Using ip_image from {args.ip_image}...")
        args.ip_image_path = args.ip_image
        args.ip_image = load_img(args.ip_image, 'RGB')

        if args.text_input is None or args.text_input == "":
            args.text_input = clip_interrogate(args.ckpt, args.ip_image, args.clip_interrogator_mode, CLIP_INTERROGATOR_MODEL_PATH)
            del_clip_interrogator_models()
            print("Using clip-interrogated prompt from ip_image as text_input:")
            print(args.text_input)

        # For now IP adapter is incompatible with LORA since they both overwrite the attention_processor for the unet
        # https://github.com/tencent-ailab/IP-Adapter/issues/69
        # TODO maybe fixable by switching to this LORA-trainer: https://github.com/kohya-ss/sd-scripts
        ip_adapter = pipe_manager.enable_ip_adapter()

        args.c, args.uc, args.pc, args.puc = ip_adapter.create_embeds(
            args.ip_image, prompt=args.text_input, negative_prompt=args.uc_text, 
            scale=args.ip_image_strength  # scale = 1.0 will mostly use the image prompt, 0.0 will only use the text prompt
            )
    else:
        print("Disabling ip_adapter..")
        pipe_manager.disable_ip_adapter()

    if args.c is None and args.text_input is not None and args.text_input != "" and 0:
                args.c, args.uc, args.pc, args.puc = pipe.encode_prompt(
                    args.text_input,
                    do_classifier_free_guidance = args.guidance_scale > 1,
                    negative_prompt = args.uc_text,
                    lora_scale = args.lora_scale,
                    )
    if 0:
        from latent_magic import sample_random_conditioning, save_ip_img_condition
        args = sample_random_conditioning(args)
        #save_ip_img_condition(args)

    if args.noise_sigma > 0.0: # apply random noise to the conditioning vectors:
        if args.c is None:
            args.c, args.uc, args.pc, args.puc = pipe.encode_prompt(
                args.text_input,
                do_classifier_free_guidance = args.guidance_scale > 1,
                negative_prompt = args.uc_text,
                lora_scale = args.lora_scale,
                )

        args_c_clone = args.c.clone()
        args_c_clone[0,1:-2,:] += torch.randn_like(args.c[0,1:-2,:]) * args.noise_sigma
        args.c = args_c_clone

    #from latent_magic import visualize_distribution
    #visualize_distribution(args, os.path.join(args.outdir, args.name))

    if args.use_lcm:
        args.guidance_scale = 0.0

    if (args.interpolator is None) and (len(args.name) == 0):
        args.name = args.text_input # send this name back to the frontend

    if (args.lora_path) and (args.interpolator is None):
        args.text_input = prepare_prompt_for_lora(args.text_input, args.lora_path, verbose = True)
        if args.c is None:
            args.c, args.uc, args.pc, args.puc = pipe.encode_prompt(
                    args.text_input,
                    do_classifier_free_guidance = args.guidance_scale > 1,
                    negative_prompt = args.uc_text,
                    lora_scale = args.lora_scale,
                    )

    if args.interpolator:
        args.seed = args.interpolator.current_seed
        args.interpolator.latent_tracker.create_new_denoising_trajectory(args, pipe)
    
    # if init image strength == 1, just return the initial image
    if (args.init_image_strength == 1.0 or (int(args.steps*(1-args.init_image_strength)) < 1)) and args.init_image and (args.controlnet_path is None):
        latent = pil_img_to_latent(args.init_image, args, _device, pipe)
        if args.interpolator:
            args.interpolator.latent_tracker.add_latent(0, pipe.scheduler.timesteps[-1], latent)

        pt_images = T.ToTensor()(args.init_image).unsqueeze(0).to(_device)
        pil_images = [args.init_image] * args.n_samples
        
        if args.upscale_f != 1.0:
            pt_images, pil_images = run_upscaler(args, pil_images)

        pil_images = maybe_apply_watermark(args, pil_images)
        return pt_images, pil_images

    if do_callback:
        callback_ = make_callback(latent_tracker = args.interpolator.latent_tracker if args.interpolator else None)
    else:
        callback_ = None

    generator = torch.Generator(device=_device).manual_seed(int(args.seed))
    
    if args.c is not None:
        prompt, prompt_2, negative_prompt = None, None, None
    else:
        prompt, prompt_2, negative_prompt = args.text_input, args.text_input_2, args.uc_text
        args.c, args.uc = None, None

    if args.n_samples > 1 and 0: # Correctly handle batches:
        prompt = [prompt] * args.n_samples
        prompt_2 = [prompt_2] * args.n_samples
        negative_prompt = [negative_prompt] * args.n_samples
        args.n_samples = 1

    if args.init_latent is not None:
        args.init_latent = args.init_latent.half()

    # SDXL is super sensitive to init_image, even with strength = 0.0, so in some cases we want to completely remove the init_img:
    if args.init_image_strength == 0.0 and args.mode == "remix":
        args.init_image = None

    denoising_start = None
    if (args.init_image is None) and (args.init_latent is not None): # lerp/real2real
        args.init_image = args.init_latent
        denoising_start = float(args.init_image_strength)
    elif (args.init_image is None) and (args.init_latent is None): # generate, no init_img
        shape = (1, pipe.unet.config.in_channels, args.H // pipe.vae_scale_factor, args.W // pipe.vae_scale_factor)
        args.init_image = torch.randn(shape, device=_device, generator=generator).half()
        args.init_image_strength = 0.0
        
    cross_attention_kwargs = {"scale": args.lora_scale} if args.lora_path else None
         
    # Common SD arguments
    pipe_fn_args = {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'image': args.init_image,
        'num_inference_steps': args.steps,
        'guidance_scale': args.guidance_scale,
        'num_images_per_prompt': args.n_samples,
        'generator': generator,
        'callback': callback_,
        'cross_attention_kwargs': cross_attention_kwargs,
        'prompt_embeds': args.c,
        'negative_prompt_embeds': args.uc,
    }

    if "XL" in str(pipe.__class__.__name__):
        pipe_fn_args.update({
            'pooled_prompt_embeds': args.pc,
            'negative_pooled_prompt_embeds': args.puc,
        })

    # Conditionally add arguments if controlnet is used
    if args.controlnet_path and args.control_image_strength > 0 and (args.control_image is not None):
        args.control_image = preprocess_controlnet_init_image(args.control_image, args)
        #args.control_image.save("control_image.png")

        pipe_fn_args.update({
            'image': args.init_image,
            'strength':  1 - args.init_image_strength,
            'control_image': args.control_image,
            'controlnet_conditioning_scale': args.control_image_strength,
            'control_guidance_start': args.control_guidance_start,
            'control_guidance_end': args.control_guidance_end,
        })
    else:
        pipe_fn_args['strength'] = 1 - args.init_image_strength
        if "XL" in str(pipe.__class__.__name__):
            pipe_fn_args.update({'denoising_start': denoising_start,})

    pipe_output = pipe(**pipe_fn_args)
    pil_images  = pipe_output.images
    pt_images   = [None]*len(pil_images)

    if args.upscale_f != 1.0:
        print(f"Upscaling with f = {args.upscale_f:.3f}...")
        pt_images, pil_images = run_upscaler(args, pil_images)

    pil_images = maybe_apply_watermark(args, pil_images)

    try:
        prompt_embeds = {key: getattr(args, key).cpu().numpy() for key in ["c", "uc", "pc", "puc"]}
    except:
        prompt_embeds = None

    free_memory, tot_mem = torch.cuda.mem_get_info(device=_device)
    print(f"End of generate, free memory: {free_memory / 1e9:.2f} GB")

    return prompt_embeds, pil_images






###########################################################################################################
###########################################################################################################








@torch.no_grad()
def make_interpolation(args, force_timepoints = None):
    # Always disbale upscaling for videos (since it introduces frame jitter)
    args.upscale_f = 1.0

    if args.interpolation_init_images and all(args.interpolation_init_images):
        mode = "real2real"
        if not args.interpolation_texts:
            args.interpolation_texts = [None]*len(args.interpolation_init_images)
    else:
        mode = "lerp"

    if mode == "real2real":
        args.controlnet_path = None
        args.init_image = None
        print("Disabling controlnet and init_image since interpolation_init_images are provided (real2real mode)")
    else: # mode == "lerp"
        if args.controlnet_path or args.init_image:
            print("Using controlnet / init_img lerp ---> disabling LatentBlending")
            args.latent_blending_skip_f = None # Disable LatentBlending with ControlNet

    if not args.interpolation_init_images:
        args.interpolation_init_images = [None]
        if args.interpolation_texts:
            args.interpolation_init_images = args.interpolation_init_images * len(args.interpolation_texts)
    if not args.interpolation_seeds:
        args.interpolation_seeds = [args.seed]
        args.n_frames = 1

    assert args.n_samples==1, "Batch size >1 not implemented for interpolation!"
    assert len(args.interpolation_texts) == len(args.interpolation_seeds), f"Number of interpolation texts ({len(args.interpolation_texts)}) does not match number of interpolation seeds ({len(args.interpolation_seeds)})"
    assert len(args.interpolation_texts) == len(args.interpolation_init_images), f"Number of interpolation texts ({len(args.interpolation_texts)}) does not match number of interpolation init images ({len(args.interpolation_init_images)})"
    assert len(args.interpolation_init_images) == len(args.interpolation_seeds), f"Number of interpolation init images ({len(args.interpolation_init_images)}) does not match number of interpolation seeds ({len(args.interpolation_seeds)})"

    if args.loop and len(args.interpolation_texts) > 2:
        args.interpolation_texts.append(args.interpolation_texts[0])
        args.interpolation_seeds.append(args.interpolation_seeds[0])
        args.interpolation_init_images.append(args.interpolation_init_images[0])

    # if there are init images, change width/height to their average
    interpolation_init_images = None
    if args.interpolation_init_images and all(args.interpolation_init_images):
        assert len(args.interpolation_init_images) == len(args.interpolation_texts), "Number of initial images must match number of prompts"
        
        interpolation_init_images = get_uniformly_sized_crops(args.interpolation_init_images, args.H * args.W)
        args.W, args.H = interpolation_init_images[0].size

        if (args.interpolation_texts is None) or len(args.interpolation_texts) == 0:
            args.interpolation_texts = [clip_interrogate(args.ckpt, init_img, args.clip_interrogator_mode, CLIP_INTERROGATOR_MODEL_PATH) for init_img in interpolation_init_images]
            print("Using clip-interrogator results:", args.interpolation_texts)
        else: # get prompts for the images that dont have one:
            assert len(args.interpolation_texts) == len(interpolation_init_images), "Number of provided prompts must match number of init_images"
            assert isinstance(args.interpolation_texts, list), "Provided interpolation_texts must be list (can contain None values where clip-interrogator is to be used)"
            for jj, init_img in enumerate(interpolation_init_images):
                if args.interpolation_texts[jj] is None:
                    init_img_prompt = clip_interrogate(args.ckpt, init_img, args.clip_interrogator_mode, CLIP_INTERROGATOR_MODEL_PATH)
                    print(f"Generated prompt for init_img_{jj}: {init_img_prompt}")
                    args.interpolation_texts[jj] = init_img_prompt

    # Load model
    pipe = pipe_manager.get_pipe(args)
    args.name = " => ".join(args.interpolation_texts) # send this name back to frontend

    # Map LORA tokens:
    if args.lora_path:
        for i, _ in enumerate(args.interpolation_texts):
            args.interpolation_texts[i] = prepare_prompt_for_lora(args.interpolation_texts[i], args.lora_path, interpolation = True, verbose = True)

    # Release CLIP memory:
    del_clip_interrogator_models()

    args.interpolator = Interpolator(
        pipe, 
        args.interpolation_texts, 
        args.n_frames, 
        args, 
        _device, 
        images = interpolation_init_images,
        smooth=args.smooth,
        seeds=args.interpolation_seeds,
        scales=[args.guidance_scale for _ in args.interpolation_texts],
        lora_paths=args.lora_paths,
    )

    n_frames  = len(args.interpolator.ts) if force_timepoints is None else len(force_timepoints)
    active_lora_path = args.lora_paths[0] if args.lora_paths else None
    interpolation_init_image = args.init_image

    ######################################

    if n_frames > 100 and False: # disable for now
        print(f"Compiling model for {args.W}x{args.H}...")
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False)

    ######################################

    for f in range(n_frames):
        force_t_raw = None
        if force_timepoints:
            force_t_raw = force_timepoints[f]

        if 1: # catch errors and try to complete the video
            try:
                t, t_raw, prompt_embeds, init_noise, scale, keyframe_index, abort_render = args.interpolator.get_next_conditioning(verbose=0, save_distances_to_dir = args.save_distances_to_dir, t_raw = force_t_raw)
            except Exception as e:
                print("Error in interpolator.get_next_conditioning(): ", str(e))
                break
        else: # get full stack_trace, for debugging:
            t, t_raw, prompt_embeds, init_noise, scale, keyframe_index, abort_render = args.interpolator.get_next_conditioning(verbose=0, save_distances_to_dir = args.save_distances_to_dir, t_raw = force_t_raw)

        if abort_render:
            return
            
        # Update all the render args for this frame:
        try: # sdxl
            args.c, args.uc, args.pc, args.puc = prompt_embeds
        except: # sdv1.x and sdv2.x
            args.c, args.uc = prompt_embeds
            args.pc, args.puc = None, None

        args.interpolator.latent_tracker.init_noises[t_raw] = init_noise
        args.guidance_scale = scale
        args.t_raw = t_raw

        # TODO, auto adjust min n_steps (needs to happend before latent blending stuff and reset after each frame render):
        # orig_n_steps = args.steps
        # args.steps = max(args.steps, int(args.min_steps/(1-args.init_image_strength)))
        # pipe.scheduler.set_timesteps(args.steps, device=_device)
        # print(f"Adjusted n_steps from {orig_n_steps} to {args.steps} to match min_steps {args.min_steps} and init_image_strength {args.init_image_strength}")
        
        if args.init_image is None and (args.latent_blending_skip_f is not None):
            args.init_latent, args.init_image, args.init_image_strength = create_init_latent(args, t, interpolation_init_images, keyframe_index, init_noise, _device, pipe)

        if args.lora_paths: # Maybe update the lora:
            if args.lora_paths[keyframe_index] != active_lora_path:
                active_lora_path = args.lora_paths[keyframe_index]
                print("Switching to lora path", active_lora_path)
                args.lora_path = active_lora_path

        #if args.planner: # When audio modulation is active:
        #    args = args.planner.adjust_args(args, t_raw, force_timepoints=force_timepoints)

        print(f"Interpolating frame {f+1}/{len(args.interpolator.ts)} "
            f"(t_raw = {t_raw:.3f}, "
            f"init_strength: {args.init_image_strength:.2f}, "
            f"latent skip_f: {args.interpolator.latent_tracker.latent_blending_skip_f:.2f}, "
            f"lpips_d: {args.interpolator.latent_tracker.frame_buffer.get_perceptual_distance_at_t(args.t_raw):.2f})"
        )
        
        # Generate the frame:
        _, pil_images = generate(args, do_callback = True)

        if args.smooth and args.latent_blending_skip_f:
            args.interpolator.latent_tracker.construct_noised_latents(args, args.t_raw)

        img_pil = pil_images[0]
        img_t = T.ToTensor()(img_pil).unsqueeze_(0).to(_device)
        args.interpolator.latent_tracker.add_frame(args, img_t, t, t_raw)

        #args.steps = orig_n_steps
        
        args.init_image = interpolation_init_image

        yield img_pil, t_raw

    # Flush the final metadata to disk if needed:
    args.interpolator.latent_tracker.reset_buffer()











###########################################################################################################
###########################################################################################################




def make_images(args):
    if args.mode == "remix" or args.mode == "upscale" or args.mode == "controlnet":
        
        if args.mode == "remix" or args.mode == "upscale":
            if args.init_image is None:
                raise ValueError(f"Must provide an init image in order to use {args.mode}!")
            img = load_img(args.init_image, 'RGB')
            w, h = img.size

            if not args.ip_image:
                print("Setting init_image as ip_image!")
                args.ip_image = args.init_image
                n_1024   = 1024 * 1024  # Number of pixels in a 1024x1024 image
                n_264    = 264 * 264    # Number of pixels in a 264x264 image
                n_pixels = w * h        # Your current image size

                # Attenuate the ip_image_strenght for low_res ip_images (since that will result in blurry imgs)
                ip_image_strength_multiplier = (n_pixels - n_264) / (n_1024 - n_264)
                ip_image_strength_multiplier = max(0.0, min(1.0, ip_image_strength_multiplier))
                args.ip_image_strength *= ip_image_strength_multiplier

        if args.mode == "upscale" and args.lora_path:
            print("Disabling LoRA for upscaling!!")
            args.lora_path = None

        if args.controlnet_path:
            if not args.control_image:
                raise ValueError(f"You must provide a control_image to use {args.mode}!")

        # remove text_input when a LoRA is active since this will trigger clip_interrogator instead of ip_adapter for now:
        if (args.mode == "remix") and args.lora_path:
            args.text_input = None
        
        if args.text_input is None or args.text_input == "" or args.text_input == "Untitled":
            init_image = load_img(args.init_image, 'RGB')
            args.text_input = clip_interrogate(args.ckpt, init_image, args.clip_interrogator_mode, CLIP_INTERROGATOR_MODEL_PATH)
            del_clip_interrogator_models()
            print("Using clip-interrogate prompt:")
            print(args.text_input)
            args.name = args.text_input
        else:
            print(f"Performing {args.mode} with provided text input: {args.text_input}")

    if args.text_input is None:
        raise ValueError(f"You must provide a text input (prompt) to use {args.mode}!")

    _, images_pil = generate(args)
    return images_pil




def run_upscaler(args_, imgs, 
        init_image_strength    = 0.55,
        upscale_guidance_scale = 6.0,
        min_upscale_steps      = 16,  # never do less than this many steps
        max_n_pixels           = 1600**2, # max number of pixels to avoid OOM
    ):
    args = copy(args_)

    # Disable all modifiers, just upscale with base SDXL model:
    args.lora_path = None
    args.controlnet_path = None

    args.W, args.H = args_.upscale_f * args_.W, args_.upscale_f * args_.H

    # set max_n_pixels to avoid OOM:
    if args.W * args.H > max_n_pixels:
        scale = math.sqrt(max_n_pixels / (args.W * args.H))
        args.W, args.H = int(scale * args.W), int(scale * args.H)

    args.W = round_to_nearest_multiple(args.W, 8)
    args.H = round_to_nearest_multiple(args.H, 8)

    x_samples_upscaled, x_images_upscaled = [], []

    # Load the upscaling model:
    #args.ckpt = args.upscale_ckpt

    if (args.c is not None) and (args.uc is not None) and args.ckpt != "sdxl-refiner-v1.0":
        args.uc_text, args.text_input = None, None
    else: # get rid of prompt conditioning vectors and just upscale with the text prompt
        args.c, args.uc, args.pc, args.puc = None, None, None, None

    free_memory, tot_mem = torch.cuda.mem_get_info(device=_device)
    print("Free memory:", free_memory / 1e9, "GB")

    if free_memory < 20e9 and (args.ckpt != args_.ckpt):
        print("Free memory is low, ready for upscaling......")
    
    upscaling_pipe = pipe_manager.get_pipe(args)

    # Avoid doing too little steps when init_image_strength is very high:
    upscale_steps = int(max(args.steps * (1-init_image_strength), min_upscale_steps) / (1-init_image_strength))+1

    for i in range(len(imgs)): # upscale in a loop:
        args.init_image = imgs[i]

        image = upscaling_pipe(
            prompt = args.text_input,
            image=args.init_image.resize((args.W, args.H)),
            guidance_scale=upscale_guidance_scale,
            strength=1-init_image_strength,
            num_inference_steps=upscale_steps,
            negative_prompt=args.uc_text,
            prompt_embeds = args.c,
            negative_prompt_embeds = args.uc,
            pooled_prompt_embeds = args.pc,
            negative_pooled_prompt_embeds = args.puc,
        ).images[0]

        x_samples_upscaled.extend([])
        x_images_upscaled.extend([image])

    return x_samples_upscaled, x_images_upscaled


def interrogate(args):
    if args.init_image is not None:
        args.init_image_path = args.init_image
        args.init_image = load_img(args.init_image, 'RGB')
    
    assert args.init_image, "Must provide an init image"
    interrogated_prompt = clip_interrogate(args.ckpt, args.init_image, args.clip_interrogator_mode, CLIP_INTERROGATOR_MODEL_PATH)
    del_clip_interrogator_models()

    return interrogated_prompt