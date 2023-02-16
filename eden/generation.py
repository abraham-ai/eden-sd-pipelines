import os
import sys
from pathlib import Path

SD_PATH = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
ROOT_PATH = SD_PATH.parents[0]
MODELS_PATH = os.path.join(ROOT_PATH, 'models')
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

from settings import *
from eden_utils import *
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionDepth2ImgPipeline
# TODO add this to the diffusers package for cleaner imports
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_eden import StableDiffusionEdenPipeline
from diffusers import LMSDiscreteScheduler, EulerDiscreteScheduler, DDIMScheduler, DPMSolverMultistepScheduler, KDPM2DiscreteScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import load_pipeline_from_original_stable_diffusion_ckpt
from diffusers.models import AutoencoderKL

from interpolator import *
#from animation import *
#from inpaint import *
#from depth import *
from clip_tools import *
from planner import LatentTracker, create_init_latent, blend_inits
from lora_diffusion import tune_lora_scale, patch_pipe


def pick_best_gpu_id():
    # pick the GPU with the most free memory:
    gpu_ids = [i for i in range(torch.cuda.device_count())]
    print(f"# of visible GPUs: {len(gpu_ids)}")
    gpu_mem = []
    for gpu_id in gpu_ids:
        free_memory, tot_mem = torch.cuda.mem_get_info(device=gpu_id)
        gpu_mem.append(free_memory)
        print("GPU %d: %d MB free" %(gpu_id, free_memory / 1024 / 1024))
    
    best_gpu_id = gpu_ids[np.argmax(gpu_mem)]
    # set this to be the active GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_id)
    print("Using GPU %d" %best_gpu_id)
    return best_gpu_id

gpu_id = pick_best_gpu_id()
_device = torch.device("cuda:%d" %gpu_id if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(_device)

# some global variables that persist between function calls:
pipe  = None
last_checkpoint = None
last_lora_path = None



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
            pipe = StableDiffusionEdenPipeline.from_pretrained(args.ckpt, safety_checker=None, torch_dtype=torch.float16 if args.half_precision else torch.float32, vae=vae)
            #pipe = StableDiffusionPipeline.from_pretrained(args.ckpt, safety_checker=None, torch_dtype=torch.float16 if args.half_precision else torch.float32, vae=vae)
    
    except Exception as e:
        print(e)
        print("Failed to load from pretrained, trying to load from checkpoint")
        pipe = load_pipeline_from_original_stable_diffusion_ckpt(args.ckpt, image_size = 512)

    set_sampler(args.sampler, pipe)
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
        args.W, args.H = match_aspect_ratio(args.W * args.H, args.init_image)
        args.init_image = args.init_image.resize((args.W, args.H), Image.LANCZOS)

    force_starting_latent = None
    if args.interpolator is not None:
        # Create a new trajectory for the latent tracker:
        args.interpolator.latent_tracker.create_new_denoising_trajectory(args)
        force_starting_latent = args.interpolator.latent_tracker.force_starting_latent
    
    # Load model
    pipe = get_pipe(args)
    
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
        extra_callback=None,
        latent_tracker = args.interpolator.latent_tracker if args.interpolator is not None else None,
    )

    generator = torch.Generator(device=_device).manual_seed(args.seed)
    generator = None

    if args.c is not None:
        prompt, negative_prompt = None, None
        #seed_everything(0)
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
            guidance_scale = args.scale,
            num_images_per_prompt = args.n_samples,
            )
    else:
        if args.init_image is not None or 1:   # img2img / Eden
            pipe_output = pipe(
                prompt = prompt,
                negative_prompt = negative_prompt, 
                width = args.W, 
                height = args.H,
                image=args.init_image, 
                strength=1-args.init_image_strength, 
                num_inference_steps = n_steps,
                guidance_scale = args.scale,
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
                guidance_scale = args.scale,
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

    pipe = get_pipe(args)
    #model = update_aesthetic_gradient_settings(model, args)
    
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
                init_img_prompts = [clip_interrogate(args.ckpt, init_img, args.clip_interrogator_mode) for init_img in interpolation_init_images]
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

    args.interpolator = Interpolator(
        pipe, 
        args.interpolation_texts, 
        args.n_frames, 
        args, 
        _device, 
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
                print("Latent blending...")
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

        _, pil_images = generate(args)
        img_pil = pil_images[0]
        img_t = T.ToTensor()(img_pil).unsqueeze_(0).to(_device)
        args.interpolator.latent_tracker.add_frame(args, img_t, t, t_raw)

        yield img_pil, t_raw

    # Flush the final metadata to disk if needed:
    args.interpolator.latent_tracker.reset_buffer()









def make_images(args, steps_per_update=None):

    if args.mode == "remix":
        enable_random_lr_flipping = True  # randomly flip the init img for remixing?

        if args.init_image_data:
            args.init_image = load_img(args.init_image_data, 'RGB')

        assert args.init_image is not None, "Must provide an init image in order to remix it!"
        
        if random.random() > 0.5 and enable_random_lr_flipping:
            args.init_image = args.init_image.transpose(Image.FLIP_LEFT_RIGHT)

        args.W, args.H = match_aspect_ratio(args.W * args.H, args.init_image)
        args.aesthetic_target = [args.init_image]
        args.text_input = clip_interrogate(args.ckpt, args.init_image, args.clip_interrogator_mode)

        del_clip_interrogator_models()

    assert args.text_input is not None

    queue = Queue() 
    job_done = object() 
    
    def callback(img, i):
        if i == args.steps-1 or i % steps_per_update == 0:
            queue.put(img)
            queue.put(None, True, timeout=None)
            queue.join()
        
    def run_make_images():
        try:
            global pipe            
            pipe = get_pipe(args)
            #pipe = update_aesthetic_gradient_settings(pipe, args)

            if steps_per_update is None:
                _, images_pil = generate(args)
                callback(images_pil, args.steps-1)
            else:
                generate(args, callback)
            queue.put(job_done)
        except Exception as exception:
            print("Exception in run_make_images: ", exception)
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



def make_callback(
    extra_callback=None,
    latent_tracker=None,
):
    # Callback for _call_ in diffusers repo, called thus:
    #   callback(i, t, latents)
    def diffusers_callback(i, t, latents):
        if latent_tracker is not None:
            latent_tracker.add_latent(latents)
        if extra_callback and 0: # TODO fix this function for discord etc...
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
    interrogated_prompt = clip_interrogate(args.ckpt, args.init_image, args.clip_interrogator_mode)
    #del_clip_interrogator_models()

    return interrogated_prompt
    