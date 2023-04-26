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

import pipe as eden_pipe
from settings import _device
from eden_utils import *
from interpolator import *
from clip_tools import *
from planner import LatentTracker, create_init_latent, blend_inits
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionDepth2ImgPipeline

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
    do_callback = False,
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
    
    # adjust min n_steps:
    #n_steps = max(args.steps, int(args.min_steps/(1-args.init_image_strength)))
    #print(f"Adjusted n_steps from {args.steps} to {n_steps} to match min_steps {args.min_steps} and init_image_strength {args.init_image_strength}")
    n_steps = args.steps

    # Load model
    pipe = eden_pipe.get_pipe(args)

    # if init image strength == 1, just return the initial image
    if (args.init_image_strength == 1.0 or (int(n_steps*(1-args.init_image_strength)) < 1)) and args.init_image:
        latent = pil_img_to_latent(args.init_image, args, _device, pipe)
        if args.interpolator is not None:
            args.interpolator.latent_tracker.add_latent(latent, init_image_strength = 1.0)

        pt_images = T.ToTensor()(args.init_image).unsqueeze(0).to(_device)
        pil_images = [args.init_image] * args.n_samples
        
        if args.upscale_f != 1.0:
            pt_images, pil_images = run_upscaler(args, pil_images)

        pil_images = maybe_apply_watermark(args, pil_images)
        return pt_images, pil_images

    # Callback
    if do_callback:
        callback_ = make_callback(
            latent_tracker = args.interpolator.latent_tracker if args.interpolator is not None else None,
            extra_callback = None,
        )
    else:
        callback_ = None

    generator = torch.Generator(device=_device).manual_seed(args.seed)
    #generator = None

    if args.c is not None:
        prompt, negative_prompt = None, None
    else:
        prompt, negative_prompt = args.text_input, args.uc_text
        args.c, args.uc = None, None

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
    
    # Always disbale upscaling for videos:
    args.upscale_f = 1.0
    
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
    pipe = eden_pipe.get_pipe(args)
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
                # get prompts for the images that dont have one:
                for jj, init_img in enumerate(interpolation_init_images):
                    if real2real_texts[jj] is None:
                        init_img_prompt = clip_interrogate(args.ckpt, init_img, args.clip_interrogator_mode, CLIP_INTERROGATOR_MODEL_PATH)
                        print(f"Generated prompt for init_img_{jj}: {init_img_prompt}")
                        real2real_texts[jj] = init_img_prompt
                args.interpolation_texts = real2real_texts

            # We're in Real2Real mode here --> overwrite args.aesthetic_target with the interpolation_init_images
            # This activates aesthetic gradient finetuning of the individual prompt conditioning vectors on each single init_image:
            #args.aesthetic_target = [[img] for img in interpolation_init_images]

    else:
        args.use_init = False

    del_clip_interrogator_models()

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
        lora_paths=args.lora_paths,
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

        if True: # catch errors and try to complete the video
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
        _, pil_images = generate(args, do_callback = True)
        img_pil = pil_images[0]
        img_t = T.ToTensor()(img_pil).unsqueeze_(0).to(_device)
        args.interpolator.latent_tracker.add_frame(args, img_t, t, t_raw)

        yield img_pil, t_raw

    # Flush the final metadata to disk if needed:
    args.interpolator.latent_tracker.reset_buffer()



class Video_Frame_Indexer():
    'Convenience class to get the correct video frame index for each prompt-to-prompt phase in the interpolation.'
    def __init__(self, n_video_frames):
        self.video_frames_per_phase = 48
        self.n_video_frames = n_video_frames
        
    def get_video_frame_index(self, t_raw):
        frame_index = int(t_raw * self.video_frames_per_phase) % self.n_video_frames
        print(f"t_raw = {t_raw:.5f}, frame_index = {frame_index}")
        return frame_index


@torch.no_grad()
def video_style_transfer(args, force_timepoints = None):
    
    # Always disbale upscaling for videos:
    args.upscale_f = 1.0

    assert (args.interpolation_texts is not None), "You must provide a sequence of prompts to use!"
    if not args.interpolation_seeds:
        args.interpolation_seeds = [args.seed]*len(args.interpolation_texts)
    assert args.n_samples==1, "Batch size >1 not implemented yet"
    assert len(args.interpolation_texts) == len(args.interpolation_seeds)

    global pipe
    pipe = eden_pipe.get_pipe(args)
    frame_indexer = Video_Frame_Indexer(len(args.interpolation_init_images))
    args.W, args.H = match_aspect_ratio(args.W*args.H, Image.open(args.interpolation_init_images[0]))

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
        lora_paths=args.lora_paths,
    )

    n_frames = len(args.interpolator.ts)
    force_timepoints = reorder_timepoints(np.linspace(0, len(args.interpolation_texts) - 1, n_frames))
    
    if force_timepoints is not None:
        n_frames = len(force_timepoints)

    active_lora_path = args.lora_paths[0] if args.lora_paths is not None else None

    ######################################

    for f in range(n_frames):
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
        
        video_init_frame_index = frame_indexer.get_video_frame_index(t_raw)

        print("Loading init image for frame", f, "from", args.interpolation_init_images[video_init_frame_index])

        args.c = c
        args.guidance_scale = scale
        args.t_raw = t_raw

        if len(args.interpolator.latent_tracker.frame_buffer.ts) < (args.interpolator.n_frames_between_two_prompts // 2):
            args.init_image = PIL.Image.open(args.interpolation_init_images[video_init_frame_index]).convert('RGB')
            args.init_latent = None
        else:
            init_img1 = PIL.Image.open(args.interpolation_init_images[video_init_frame_index]).convert('RGB')
            init_img2 = init_img1
            args.init_latent, args.init_image, args.init_image_strength = create_init_latent(args, t, init_img1, init_img2, _device, pipe, real2real = True)


        if args.lora_paths is not None:
            # Maybe update the lora to the next person:
            if args.lora_paths[return_index] != active_lora_path:
                active_lora_path = args.lora_paths[return_index]
                print("Switching to lora path", active_lora_path)
                args.lora_path = active_lora_path

        if args.planner is not None: # When audio modulation is active:
            args = args.planner.adjust_args(args, t_raw, force_timepoints=force_timepoints)

        print(f"Interpolating frame {f+1}/{len(args.interpolator.ts)} (t_raw = {t_raw:.5f},\
                init_strength: {args.init_image_strength:.2f},\
                latent skip_f: {args.interpolator.latent_tracker.latent_blending_skip_f:.2f},\
                splitting lpips_d: {args.interpolator.latent_tracker.frame_buffer.get_perceptual_distance_at_t(args.t_raw):.2f}),\
                keyframe {return_index+1}/{len(args.interpolation_texts) - 1}...")


        args.lora_path = active_lora_path
        _, pil_images = generate(args, do_callback = True)
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

from pipe import set_sampler

def run_upscaler(args_, imgs, init_image_strength = 0.68, upscale_steps = 25, upscale_guidance_scale = 6.5):
    args = copy(args_)
    args.W, args.H = args_.upscale_f * args_.W, args_.upscale_f * args_.H
    args.W = round_to_nearest_multiple(args.W, 64)
    args.H = round_to_nearest_multiple(args.H, 64)

    x_samples_upscaled, x_images_upscaled = [], []

    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(args.ckpt, torch_dtype=torch.float16)
    pipe_img2img = pipe_img2img.to(_device)
    pipe_img2img.enable_xformers_memory_efficient_attention()

    set_sampler("euler", pipe_img2img)

    for i in range(len(imgs)): # upscale in a loop:
        args.init_image = imgs[i]
        image = pipe_img2img(
            args.text_input,
            image=args.init_image.resize((args.W, args.H)),
            guidance_scale=upscale_guidance_scale,
            strength=1-init_image_strength,
            num_inference_steps=upscale_steps,
            negative_prompt=args.uc_text,
        ).images[0]
        x_samples_upscaled.extend([])
        x_images_upscaled.extend([image])

    return x_samples_upscaled, x_images_upscaled


def interrogate(args):
    if args.init_image_data:
        args.init_image = load_img(args.init_image_data, 'RGB')
    
    assert args.init_image is not None, "Must provide an init image"
    interrogated_prompt = clip_interrogate(args.ckpt, args.init_image, args.clip_interrogator_mode, CLIP_INTERROGATOR_MODEL_PATH)
    #del_clip_interrogator_models()

    return interrogated_prompt
    