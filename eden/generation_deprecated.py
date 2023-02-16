import os
import sys
from pathlib import Path
SD_PATH = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
ROOT_PATH = SD_PATH.parents[0]
KD_PATH = os.path.join(ROOT_PATH, 'k-diffusion')
MIDAS_PATH = os.path.join(ROOT_PATH, 'MiDaS')
ADABINS_PATH = os.path.join(ROOT_PATH, 'AdaBins')
FILM_PATH = os.path.join(ROOT_PATH, 'frame-interpolation')
CLIP_INTERROGATOR_PATH = os.path.join(ROOT_PATH, 'clip-interrogator')
CLIP_INTERROGATOR_MODEL_PATH = os.path.join(ROOT_PATH, 'cache')
MODELS_PATH = os.path.join(ROOT_PATH, 'models')
sys.path.append(KD_PATH)
sys.path.append(MIDAS_PATH)
sys.path.append(ADABINS_PATH)
sys.path.append(FILM_PATH)
sys.path.append(CLIP_INTERROGATOR_PATH)

from _thread import start_new_thread
from queue import Queue
from copy import copy
import math
import cv2
import numpy as np
import random
import requests
from PIL import Image
from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
from torch import autocast
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from settings import *
from eden_utils import *

#from depth import *
#from interpolator import *
#from animation import *
#from inpaint import *
#from clip_tools import *

#from planner import LatentTracker, create_init_latent, blend_inits

def maybe_apply_watermark(args, x_images):
    # optionally, apply watermark to final image:
    if args.watermark_path is not None:
        # check if args.watermarker already exists:
        if not hasattr(args, 'watermarker'):
            args.watermarker = WaterMarker(x_images.shape[2], x_images.shape[1], args.watermark_path) 
        # apply watermark:
        x_images = args.watermarker.apply_watermark(x_images)
    return x_images

@torch.no_grad()
def generate(
    args, 
    callback=None,
    upscale = False,
):
    assert args.text_input is not None
    seed_everything(args.seed)

    args.W = round_to_nearest_multiple(args.W, 64)
    args.H = round_to_nearest_multiple(args.H, 64)

    print("HEREEEEEE")
    print("HEREEEEEE")
    print("HEREEEEEE")

    # Load model
    model = get_model(args.config, args.ckpt, args.half_precision)
    model_wrap = CompVisDenoiser(model) 
    batch_size = args.n_samples

    if args.interpolator is not None:
        # Create a new trajectory for the latent tracker:
        args.interpolator.latent_tracker.create_new_denoising_trajectory(args)

    # Load init image
    if args.init_image_data:
        args.init_image = load_img(args.init_image_data, 'RGB')

    # if init image strength == 1, just return the initial image
    if args.init_image_strength == 1.0 and args.init_image:
        latent = pil_img_to_latent(args.init_image, args, device, model, batch_size = batch_size)
        if args.interpolator is not None:
            args.interpolator.latent_tracker.add_latent(latent, init_image_strength = 1.0)

        x_samples = T.ToTensor()(args.init_image).unsqueeze(0).to(device)
        x_images = 255. * rearrange(x_samples, 'b c h w -> b h w c')
        
        if args.upscale_f != 1.0:
            x_samples, x_images = run_upscaler(args, x_images)

        x_images = maybe_apply_watermark(args, x_images)
        return x_samples, x_images

    # get the denoising schedule:
    n_steps = max(args.steps, int(args.min_steps/(1-args.init_image_strength)))
    k_sigmas, t_enc = get_k_sigmas(model_wrap, args.init_image_strength, n_steps)

    # Load mask image
    if args.mask_image_data:
        args.mask_image = load_img(args.mask_image_data, 'L')

    args.use_init = args.init_image is not None
    args.use_mask = args.mask_image and args.init_image_strength<1

    # Initial image
    init_latent = None

    force_starting_latent = None
    if args.interpolator is not None:
        force_starting_latent = args.interpolator.latent_tracker.force_starting_latent

    if force_starting_latent is None:
        if args.init_latent is not None:
            init_latent = args.init_latent
            args.use_init = True
        elif args.init_sample is not None: # same as init_image, but already a PyTorch img tensor
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(args.init_sample))
        elif args.use_init and args.init_image is not None and args.init_image != '':
            if args.use_mask and args.init_image_inpaint_mode:
                args.init_image = inpaint_init_image(args.init_image, args.mask_image, args.init_image_inpaint_mode)
            init_latent = pil_img_to_latent(args.init_image, args, device, model, batch_size = batch_size)

    # Mask 
    mask = None
    if args.use_mask:
        assert args.mask_image is not None, "use_mask==True: A mask image is required for a mask"
        assert args.use_init, "use_mask==True: use_init is required for a mask"
        assert init_latent is not None, "use_mask==True: An latent init image is required for a mask"

        if args.mask_image_data and not args.mask_image:
            args.mask_image = load_img(args.mask_image_data, 'RGBA')

        mask = prepare_mask(args.mask_image, 
                            init_latent.shape, 
                            args.mask_contrast_adjust, 
                            args.mask_brightness_adjust,
                            args.mask_invert)        
        mask = mask.to(device)
        mask = repeat(mask, '1 ... -> b ...', b=batch_size)

    # Sampler
    if args.sampler == 'plms':
        sampler = PLMSSampler(model)
    elif args.sampler == 'dpm':
        sampler = DPMSolverSampler(model)
    else:
        sampler = DDIMSampler(model)

    if args.sampler in ['plms','ddim']:
        sampler.make_schedule(ddim_num_steps=n_steps, ddim_eta=args.ddim_eta, verbose=False)

    # Callback
    callback_ = make_callback(
        model=model,
        sampler_name=args.sampler,
        dynamic_threshold=args.dynamic_threshold, 
        static_threshold=args.static_threshold,
        mask=mask, 
        init_latent=init_latent,
        sigmas=k_sigmas,
        sampler=sampler,
        extra_callback=callback,
        latent_tracker = args.interpolator.latent_tracker if args.interpolator is not None else None,
    )

    precision_scope = autocast if args.precision == "autocast" else nullcontext

    with precision_scope("cuda"):
        with model.ema_scope():
            uc = args.uc if args.scale != 1.0 else None

            if args.sampler in ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral"]:
                samples = sampler_fn(
                    c=args.c, 
                    uc=args.uc, 
                    args=args, 
                    model_wrap=model_wrap, 
                    init_latent=init_latent, 
                    x=force_starting_latent,
                    t_enc=t_enc, 
                    device=device, 
                    cb=callback_,
                    sigmas=k_sigmas,
                    noise=create_seeded_noise(args.seed, args, device, batch_size))
            else:
                print("Sampling with", args.sampler, "sampler")
                print("This is not recommended since these sampler function have not really been maintained with other features of the repo...")
                if init_latent is not None and args.init_image_strength > 0:
                    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                else:
                    z_enc = torch.randn([args.n_samples, args.C, args.H // args.f, args.W // args.f], device=device)
                if args.sampler == 'ddim':
                    samples = sampler.decode(
                        z_enc, 
                        args.c, 
                        t_enc, 
                        unconditional_guidance_scale=args.scale,
                        unconditional_conditioning=args.uc,
                        img_callback=callback_
                    )
                elif args.sampler == 'plms': 
                    shape = [args.C, args.H // args.f, args.W // args.f]
                    samples, _ = sampler.sample(
                        S=n_steps,
                        conditioning=args.c,
                        batch_size=args.n_samples,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=args.scale,
                        unconditional_conditioning=args.uc,
                        eta=args.ddim_eta,
                        x_T=z_enc,
                        img_callback=callback_
                    )
                elif args.sampler == 'dpm':
                    samples,_ = sampler.sample(
                        S=n_steps,
                        conditioning=args.c,
                        batch_size=args.n_samples,
                        shape=[args.C, args.H // args.f, args.W // args.f],
                        verbose=True,
                        unconditional_guidance_scale=args.scale,
                        unconditional_conditioning=args.uc,
                        eta=args.ddim_eta,
                        x_T=z_enc,
                        img_callback=callback_,                            
                    )
                else:
                    raise Exception(f"Sampler {args.sampler} not recognised.")

            if args.interpolator is not None: # add the final denoised latent to the tracker:
                args.interpolator.latent_tracker.add_latent(samples, init_image_strength = args.init_image_strength)

            x_samples = model.decode_first_stage(samples, force_not_quantize=True)
            x_images = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            x_images = 255. * rearrange(x_images, 'b c h w -> b h w c')
            
            if args.upscale_f != 1.0:
                print(f"Upscaling with f = {args.upscale_f:.3f}...")
                x_samples, x_images = run_upscaler(args, x_images)

            x_images = maybe_apply_watermark(args, x_images)
            return x_samples, x_images


def interrogate(args):

    if args.init_image_data:
        args.init_image = load_img(args.init_image_data, 'RGB')
    
    assert args.init_image is not None, "Must provide an init image"
    interrogated_prompt = clip_interrogate(args.ckpt, args.init_image, args.clip_interrogator_mode, CLIP_INTERROGATOR_MODEL_PATH)
    #del_clip_interrogator_models()

    return interrogated_prompt


def make_images(args, steps_per_update=None):

    if args.mode == "remix":
        enable_random_lr_flipping = False  # randomly flip the init img for remixing?

        if args.init_image_data:
            args.init_image = load_img(args.init_image_data, 'RGB')

        assert args.init_image is not None, "Must provide an init image"
        
        if random.random() > 0.5 and enable_random_lr_flipping:
            args.init_image = args.init_image.transpose(Image.FLIP_LEFT_RIGHT)

        args.W, args.H = match_aspect_ratio(args.W * args.H, args.init_image)
        args.aesthetic_target = [args.init_image]
        args.text_input = clip_interrogate(args.ckpt, args.init_image, args.clip_interrogator_mode, CLIP_INTERROGATOR_MODEL_PATH)

    assert args.text_input is not None
    #assert args.upscale_f == 1.0 or args.n_samples == 1, "Upscaling not supported for batched generation yet"

    #del_clip_interrogator_models()

    queue = Queue() 
    job_done = object() 
    
    def callback(img, i):
        if i == args.steps-1 or i % steps_per_update == 0:
            queue.put(img)
            queue.put(None, True, timeout=None)
            queue.join()
        
    def run_make_images():

        global model
        model = get_model(args.config, args.ckpt, args.half_precision)
        model = update_aesthetic_gradient_settings(model, args)
        args.uc = get_prompt_conditioning([args.uc_text] * args.n_samples, args.precision)
        args.c  = get_prompt_conditioning([args.text_input] * args.n_samples, args.precision)

        if steps_per_update is None:
            _, images = generate(args)
            images_pil = [Image.fromarray(img.cpu().numpy().astype(np.uint8)) for img in images]
            callback(images_pil, args.steps-1)
        else:
            generate(args, callback)
        queue.put(job_done)


        try:            
            global model
            model = get_model(args.config, args.ckpt, args.half_precision)
            model = update_aesthetic_gradient_settings(model, args)
            args.uc = get_prompt_conditioning([args.uc_text] * args.n_samples, args.precision)
            args.c  = get_prompt_conditioning([args.text_input] * args.n_samples, args.precision)

            if steps_per_update is None:
                _, images = generate(args)
                images_pil = [Image.fromarray(img.cpu().numpy().astype(np.uint8)) for img in images]
                callback(images_pil, args.steps-1)
            else:
                generate(args, callback)
            queue.put(job_done)
        except Exception as exception:
            queue.put(exception)
            queue.put(job_done)
            raise exception

    start_new_thread(run_make_images, ())
    frame_idx = 0
    while True:
        next_item = queue.get(True, timeout=None)
        queue.task_done()
        if next_item is job_done:
            break
        yield next_item, frame_idx
        frame_idx += 1
        queue.get()
        queue.task_done()


def make_interpolation(args, force_timepoints = None):
    
    #if force_timepoints is not None: # When forcing the timepoints, we don't want to enable smoothing:
    #    args.smooth = False

    if args.sampler == "euler_ancestral":
        print("WARNING: euler_ancestral sampler is not great for interpolation...")
        print("Consider using another sampler to get smoother videos!")

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

    global model
    model = get_model(args.config, args.ckpt, args.half_precision)    
    args.uc = get_prompt_conditioning(args.uc_text, args.precision)
    model = update_aesthetic_gradient_settings(model, args)
    args.use_mask = False

    # if there are init images, change width/height to their average
    interpolation_init_images = None
    if args.interpolation_init_images and all(args.interpolation_init_images):
        assert len(args.interpolation_init_images) == len(args.interpolation_texts), "Number of initial images must match number of prompts"
        
        args.use_init = True
        interpolation_init_images = get_uniformly_sized_crops(args.interpolation_init_images, args.H * args.W)
        args.W, args.H = interpolation_init_images[0].size

        # if args.interpolation_init_images_use_img2txt, then use prompt-search img2txt to overwrite interpolation_texts
        if args.interpolation_init_images_use_img2txt:

            if real2real_texts is None:
                try:
                    init_img_prompts = [clip_interrogate(args.ckpt, init_img, args.clip_interrogator_mode, CLIP_INTERROGATOR_MODEL_PATH) for init_img in interpolation_init_images]
                    args.interpolation_texts = init_img_prompts
                    print("Overwriting prompts with clip-interrogator results:", init_img_prompts)

                except Exception as e:
                    print(f"Failed to get prompts from initial images. Reason: {e}. Falling back to original text prompts.")

            else:
                args.interpolation_texts = real2real_texts

            # We're in Real2Real mode here --> overwrite args.aesthetic_target with the interpolation_init_images
            # This activates aesthetic gradient finetuning of the individual prompt conditioning vectors on each single init_image:
            args.aesthetic_target = [[img] for img in interpolation_init_images]

    else:
        args.use_init = False

    del_clip_interrogator_models()

    args.interpolator = Interpolator(
        model, 
        args.interpolation_texts, 
        args.n_frames, 
        args, 
        device, 
        smooth=args.smooth,
        seeds=args.interpolation_seeds,
        scales=[args.scale for _ in args.interpolation_texts],
        scale_modulation_amplitude_multiplier=args.scale_modulation,
    )

    #args.n_anchor_imgs = max(3, int(args.anchor_img_fraction * args.interpolator.n_frames_between_two_prompts))
    #args.n_anchor_imgs = min(args.n_anchor_imgs, 7)
    print("Using", args.n_anchor_imgs, "anchor images per prompt pair")

    n_frames = len(args.interpolator.ts)
    if force_timepoints is not None:
        n_frames = len(force_timepoints)

    ######################################

    for f in range(n_frames):
        print("----------------------------------------------------------------------------------------------------------------")
        if force_timepoints is not None:
            force_t_raw = force_timepoints[f]
        else:
            force_t_raw = None

        if 0: # catch errors and try to complete the video
            try:
                t, t_raw, c, init_latent, scale, return_index, _, _ = args.interpolator.get_next_conditioning(verbose=0, save_distances_to_dir = args.save_distances_to_dir, t_raw = force_t_raw)
            except Exception as e:
                print("Error in interpolator.get_next_conditioning(): ", str(e))
                break
        else: # get full stack_trace, for debugging:
            t, t_raw, c, init_latent, scale, return_index, _, _ = args.interpolator.get_next_conditioning(verbose=0, save_distances_to_dir = args.save_distances_to_dir, t_raw = force_t_raw)
        
        args.c = c
        args.scale = scale
        args.t_raw = t_raw

        if args.interpolation_init_images and all(args.interpolation_init_images) or len(args.interpolator.latent_tracker.frame_buffer.ts) >= args.n_anchor_imgs:

            if interpolation_init_images is None: # lerping mode (no init imgs)
                is_real2real = False
                init_img1, init_img2 = args.interpolator.latent_tracker.frame_buffer.get_current_keyframe_imgs()
                init_img1, init_img2 = sample_to_pil(init_img1), sample_to_pil(init_img2)
            else: # real2real mode
                is_real2real = True
                init_img1, init_img2 = interpolation_init_images[return_index], interpolation_init_images[return_index + 1]
            
            if len(args.interpolator.latent_tracker.frame_buffer.ts) < args.n_anchor_imgs and 0:
                print("Pixel blending...")
                # apply linear blending of keyframe images in pixel space and then encode
                args.init_image, args.init_image_strength = blend_inits(init_img1, init_img2, t, args, real2real = is_real2real)
                args.init_latent = None
            else:
                # perform Latent-Blending initialization:
                args.init_latent, args.init_image, args.init_image_strength = create_init_latent(args, t, init_img1, init_img2, device, model, real2real = is_real2real)

        else: #only use the raw init_latent noise from interpolator (using the input seeds)
            args.init_latent = init_latent
            args.init_image, args.init_image_strength = None, 0.0

        if args.planner is not None: # When audio modulation is active:
            args = args.planner.adjust_args(args, t_raw, force_timepoints=force_timepoints)

        
        print(f"Interpolating frame {f+1}/{len(args.interpolator.ts)} (t_raw = {t_raw:.5f},\
 latent skip_f: {args.interpolator.latent_tracker.latent_blending_skip_f:.3f},\
 splitting lpips_d: {args.interpolator.latent_tracker.frame_buffer.get_perceptual_distance_at_t(args.t_raw):.3f}),\
 keyframe {return_index+1}/{len(args.interpolation_texts) - 1}...")

        _, new_images = generate(args)
        img_pil = Image.fromarray(new_images[0].cpu().numpy().astype(np.uint8))
        img_t = T.ToTensor()(img_pil).unsqueeze_(0).to(device)

        args.interpolator.latent_tracker.add_frame(args, img_t, t, t_raw)

        yield img_pil, t_raw

    # Flush the final metadata to disk if needed:
    args.interpolator.latent_tracker.reset_buffer()

def make_callback(
    model,
    sampler_name, 
    dynamic_threshold=None, 
    static_threshold=None, 
    mask=None, 
    init_latent=None, 
    sigmas=None, 
    sampler=None, 
    masked_noise_modifier=1.0,
    extra_callback=None,
    latent_tracker=None,
):
    # Creates the callback function to be passed into the samplers
    # The callback function is applied to the image at each step
    def dynamic_thresholding_(img, threshold):
        # Dynamic thresholding from Imagen paper (May 2022)
        s = np.percentile(np.abs(img.cpu()), threshold, axis=tuple(range(1,img.ndim)))
        s = np.max(np.append(s,1.0))
        torch.clamp_(img, -1*s, s)
        torch.FloatTensor.div_(img, s)

    # Callback for samplers in the k-diffusion repo, called thus:
    #   callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
    def k_callback_(args_dict):
        if latent_tracker is not None:
            latent_tracker.add_latent(args_dict['x'])
        if dynamic_threshold is not None:
            dynamic_thresholding_(args_dict['x'], dynamic_threshold)
        if static_threshold is not None:
            torch.clamp_(args_dict['x'], -1*static_threshold, static_threshold)
        if mask is not None:
            init_noise = init_latent + noise * args_dict['sigma']
            is_masked = torch.logical_and(mask >= mask_schedule[args_dict['i']], mask != 0 )
            new_img = init_noise * torch.where(is_masked,1,0) + args_dict['x'] * torch.where(is_masked,0,1)
            args_dict['x'].copy_(new_img)
        if extra_callback:
            x = model.decode_first_stage(args_dict['x'])
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255. * rearrange(x, 'b c h w -> b h w c')
            x = x.cpu().numpy().astype(np.uint8)
            x = [Image.fromarray(x_) for x_ in x]
            extra_callback(x, args_dict['i'])

    # Function that is called on the image (img) and step (i) at each step
    def img_callback_(img, i):
        # Thresholding functions
        if dynamic_threshold is not None:
            dynamic_thresholding_(img, dynamic_threshold)
        if static_threshold is not None:
            torch.clamp_(img, -1*static_threshold, static_threshold)
        if mask is not None:
            i_inv = len(sigmas) - i - 1
            init_noise = sampler.stochastic_encode(init_latent, torch.tensor([i_inv]*batch_size).to(device), noise=noise)
            is_masked = torch.logical_and(mask >= mask_schedule[i], mask != 0 )
            new_img = init_noise * torch.where(is_masked,1,0) + img * torch.where(is_masked,0,1)
            img.copy_(new_img)
        if extra_callback:
            extra_callback(img, i)
              
    if init_latent is not None:
        noise = torch.randn_like(init_latent, device=device) * masked_noise_modifier
    if sigmas is not None and len(sigmas) > 0:
        mask_schedule, _ = torch.sort(sigmas/torch.max(sigmas))
    elif len(sigmas) == 0:
        mask = None # no mask needed if no steps (usually happens because strength==1.0)
    if sampler_name in ["plms","ddim"]: 
        # Callback function formated for compvis latent diffusion samplers
        if mask is not None:
            assert sampler is not None, "Callback function for stable-diffusion samplers requires sampler variable"
            batch_size = init_latent.shape[0]

        callback = img_callback_
    else: 
        # Default callback function uses k-diffusion sampler variables
        callback = k_callback_

    return callback



def make_animation(args_):
    args = copy(args_)
    assert args.n_samples==1, "Batch size >1 not implemented yet"
    assert not (args.animation_mode is None and args.mode == 'animate')
    
    model = get_model(args.config, args.ckpt, args.half_precision)
    args.uc = get_prompt_conditioning(args.uc_text, args.precision)
    model = update_aesthetic_gradient_settings(model, args)

    args.use_init = args.init_image is not None or args.init_video is not None
    args.use_mask = args.mask_image and args.init_image_strength<1

    if not args.interpolation_texts:
        args.interpolation_texts = [args.text_input]
    if not args.interpolation_seeds:
        args.interpolation_seeds = [args.seed]
        args.n_frames = 1

    anim_args, keys = get_anim_args(args)
    using_vid_init = args.init_video is not None    
    if anim_args.animation_mode == '3D':
        setup_depth_models(MODELS_PATH)
    
    # state for interpolating between diffusion steps
    turbo_steps = 1 if using_vid_init else int(anim_args.diffusion_cadence)
    turbo_prev_image, turbo_prev_frame_idx = None, 0
    turbo_next_image, turbo_next_frame_idx = None, 0

    images = list()
    prev_sample = None
    color_match_sample = None
    frame_idx = 0

    while frame_idx < anim_args.max_frames:
        print(f"Rendering animation frame {frame_idx} of {anim_args.max_frames}")
        noise = keys.noise_schedule_series[frame_idx]
        strength = keys.strength_schedule_series[frame_idx]
        contrast = keys.contrast_schedule_series[frame_idx]
        
        # emit in-between frames
        if turbo_steps > 1:
            tween_frame_start_idx = max(0, frame_idx-turbo_steps)
            for tween_frame_idx in range(tween_frame_start_idx, frame_idx):
                tween = float(tween_frame_idx - tween_frame_start_idx + 1) / float(frame_idx - tween_frame_start_idx)
                print(f"  creating in between frame {tween_frame_idx} tween:{tween:0.2f}")
                if anim_args.animation_mode == '2D':
                    if turbo_prev_image is not None and tween_frame_idx > turbo_prev_frame_idx:
                        turbo_prev_image = anim_frame_warp_2d(turbo_prev_image, args, anim_args, keys, tween_frame_idx)
                    if tween_frame_idx > turbo_next_frame_idx:
                        turbo_next_image = anim_frame_warp_2d(turbo_next_image, args, anim_args, keys, tween_frame_idx)
                else: # '3D'
                    if turbo_prev_image is not None and tween_frame_idx > turbo_prev_frame_idx:
                        prev_depth = predict_depth(turbo_prev_image, anim_args)
                        turbo_prev_image = anim_frame_warp_3d(turbo_prev_image, prev_depth, anim_args, keys, tween_frame_idx)
                    if tween_frame_idx > turbo_next_frame_idx:
                        next_depth = predict_depth(turbo_next_image, anim_args)
                        turbo_next_image = anim_frame_warp_3d(turbo_next_image, next_depth, anim_args, keys, tween_frame_idx)
                turbo_prev_frame_idx = turbo_next_frame_idx = tween_frame_idx

                if turbo_prev_image is not None and tween < 1.0:
                    img = turbo_prev_image*(1.0-tween) + turbo_next_image*tween
                else:
                    img = turbo_next_image

                img_pil = Image.fromarray(img.astype(np.uint8))
                #images.append(img_pil)                
                yield img_pil, frame_idx

            if turbo_next_image is not None:
                prev_sample = sample_from_cv2(turbo_next_image)

        # apply transforms to previous frame
        if prev_sample is not None:
            if anim_args.animation_mode == '2D':
                prev_img = anim_frame_warp_2d(sample_to_cv2(prev_sample), args, anim_args, keys, frame_idx)
            else: # '3D'
                prev_img_cv2 = sample_to_cv2(prev_sample)
                depth = predict_depth(prev_img_cv2, anim_args)
                prev_img = anim_frame_warp_3d(prev_img_cv2, depth, anim_args, keys, frame_idx)

            # apply color matching
            if anim_args.color_coherence is not None:
                if color_match_sample is None:
                    color_match_sample = prev_img.copy()
                else:
                    prev_img = maintain_colors(prev_img, color_match_sample, anim_args.color_coherence)

            # apply scaling
            contrast_sample = prev_img * contrast
            # apply frame noising
            noised_sample = add_noise(sample_from_cv2(contrast_sample), noise)

            # use transformed previous frame as init for current
            args.use_init = True
            if args.half_precision:
                args.init_sample = noised_sample.half().to(device)
            else:
                args.init_sample = noised_sample.to(device)
            args.init_image_strength = max(0.0, min(1.0, strength))

        # grab prompt for current frame
        args.prompt = args.text_input #prompt_series[frame_idx]
        print(f"{args.prompt} {args.seed}")

        # grab init image for current frame
        if using_vid_init:
            init_frame = os.path.join(args.init_video, f"{frame_idx+1:04}.jpg")
            print(f"Using video init frame {init_frame}")
            args.init_image_data = init_frame

        # sample the diffusion model
        args.c = get_prompt_conditioning(args.text_input, args.precision)
        sample, image = generate(args)

        if not using_vid_init:
            prev_sample = sample

        if turbo_steps > 1:
            turbo_prev_image, turbo_prev_frame_idx = turbo_next_image, turbo_next_frame_idx
            turbo_next_image, turbo_next_frame_idx = sample_to_cv2(sample, type=np.float32), frame_idx
            frame_idx += turbo_steps
        else:   
            img_pil = Image.fromarray(image[0].cpu().numpy().astype(np.uint8))
            #images.append(img_pil)
            frame_idx += 1
            yield img_pil, frame_idx

        args.seed = args.seed + 1
        # args.seed = next_seed(args)


def run_upscaler(args_, imgs, init_image_strength = 0.7, min_steps = 30):
    args = copy(args_)
    args.n_samples = 1  # batching will prob cause OOM, so run everything in a loop
    args.init_image_data = None
    args.init_latent = None
    #args.interpolator.latent_tracker = None
    args.init_image_strength = init_image_strength
    args.steps = int(min_steps/(1-args.init_image_strength))
    args.W, args.H = args_.upscale_f * args_.W, args_.upscale_f * args_.H
    args.upscale_f = 1.0  # don't upscale again

    x_samples_upscaled, x_images_upscaled = [], []

    for i in range(imgs.shape[0]): # upscale in a loop:
        args.init_image = Image.fromarray(imgs[i].cpu().numpy().astype(np.uint8))
        args.c  = args_.c[i].unsqueeze(0)
        args.uc = args_.uc[i].unsqueeze(0)
        x_samples, x_images = generate(args)
        x_samples_upscaled.append(x_samples)
        x_images_upscaled.append(x_images)

    x_samples = torch.cat(x_samples_upscaled, dim=0)
    x_images  = torch.cat(x_images_upscaled, dim=0)

    return x_samples, x_images