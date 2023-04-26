import sys
sys.path.append('..')

import json
import time
import os
import random
from PIL import Image
import numpy as np
import moviepy.editor as mpy

from settings import StableDiffusionSettings
from generation import *
from prompts import text_inputs
from eden_utils import *

def remix(init_image, prompt, upscale_init_strength, target_n_pixels, steps, img_basename, outdir, 
    seed = int(time.time())):

    args = StableDiffusionSettings(
        mode = "remix",
        clip_interrogator_mode = "fast",
        text_input=prompt,
        init_image = init_image,
        W = int(np.sqrt(target_n_pixels)//64 * 64),
        H = int(np.sqrt(target_n_pixels)//64 * 64),
        sampler = "euler",
        guidance_scale = 7,
        seed = seed,
        n_samples = 1,
        upscale_f = 1.0,
        init_image_strength = upscale_init_strength
    )

    args.W, args.H = match_aspect_ratio(args.W * args.H, args.init_image)

    # Run the upscaler:
    _, imgs = run_upscaler(args, [args.init_image], init_image_strength = upscale_init_strength, upscale_steps = steps)

    name = f'{img_basename}_remix_{upscale_init_strength:.2f}_{int(time.time())}'
    for i, img in enumerate(imgs):
        frame = f'{name}_{i}.jpg'
        os.makedirs(outdir, exist_ok = True)
        img.save(os.path.join(outdir, frame), quality=95)

    # Also save the original image:
    args.init_image.save(os.path.join(outdir, f'{img_basename}_orig.jpg'), quality=95)

    if 1:
        # save settings
        settings_filename = f'{outdir}/{name}.json'
        save_settings(args, settings_filename)

def get_frame_paths(input_dir, phase_i, n_frames_per_phase):
    frame_paths = []
    #TODO
    return frame_paths


if __name__ == "__main__":

    # IO settings:
    outdir = "results/upscaling"
    input_dir = "/home/rednax/SSD2TB/Github_repos/cog/eden-sd-pipelines/eden/xander/waking_life/real2real_seed_1__pass_0__1_168236895012"

    # upscale settings:
    steps                = 60
    init_strength        = 0.5
    base_target_n_pixels = 1920*1080

    phase_data_dir = os.path.join(input_dir, "phase_data")
    if os.path.exists(phase_data_dir):
        phase_datapaths = sorted([f for f in os.listdir(phase_data_dir) if f.endswith('.npz')])
        n_phases = len(phase_datapaths)
        video_args = json.load(open(os.path.join(input_dir, "args.json")))
    else:
        # Upscaling settings:
        clip_interrogator_modes = ["fast"]


    for phase_i in range(n_phases):
        frames_paths = get_frame_paths(input_dir, phase_i, video_args['n_frames_per_phase'])

        # Load the phase data:
        phase_data = np.load(os.path.join(phase_data_dir, phase_datapaths[phase_i]))

        # Load the low-res frames:
        low_res_frames   = [load_img(path, 'RGB') for path in frames_paths]

        # encode all the frames into the latent space:    

        # run a hacked real2real where the init_latent at every step is the corresponding encoded frame + some noise
        
