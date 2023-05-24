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

def remix(init_image, prompt, upscale_init_strength, target_n_pixels, steps, img_basename, outdir, checkpoint,
    seed = int(time.time())):

    args = StableDiffusionSettings(
        mode = "remix",
        ckpt = checkpoint,
        clip_interrogator_mode = "fast",
        text_input=prompt,
        init_image = init_image,
        steps = steps,
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
    _, imgs = run_upscaler(args, [args.init_image], init_image_strength = upscale_init_strength, min_upscale_steps = 20)

    name = f'{img_basename}_remix_{upscale_init_strength:.2f}_{os.path.basename(checkpoint)}_timestamp_{int(time.time())}'
    for i, img in enumerate(imgs):
        if i>0:
            frame = f'{name}_{i}.jpg'
        else:
            frame = f'{name}.jpg'
        os.makedirs(outdir, exist_ok = True)
        img.save(os.path.join(outdir, frame), quality=95)

    # Also save the original image:
    args.init_image.save(os.path.join(outdir, f'{img_basename}_orig.jpg'), quality=95)

    if 1:
        # save settings
        settings_filename = f'{outdir}/{name}.json'
        save_settings(args, settings_filename)


if __name__ == "__main__":

    # IO settings:
    outdir = "results/upscaled"
    init_image_data = "../assets"

    # Upscaling settings:
    checkpoint_options      = ["eden:eden-v1", "dreamlike-art/dreamlike-photoreal-2.0"]
    clip_interrogator_modes = ["fast", "full"]
    steps                   = 80
    init_strengths_per_img  = [0.4, 0.6]
    base_target_n_pixels    = 1920*1080
    
    try_to_load_prompts_from_disk = True  # if False, always use CLIP_INTERROGATOR

    img_extensions = ['.jpg', '.png', '.jpeg']

    ###########################################################

    if os.path.isdir(init_image_data):
        # recursively grab all imgs in directory:
        init_image_data = sorted([os.path.join(init_image_data, f) for f in os.listdir(init_image_data) if os.path.splitext(f)[1].lower() in img_extensions])
    else:
        # assume it's a single image:
        init_image_data = [init_image_data]

    for init_img_data in init_image_data:
        init_image   = load_img(init_img_data, 'RGB')
        img_basename = os.path.splitext(os.path.basename(init_img_data))[0]

        prompts = None
        if try_to_load_prompts_from_disk:
            json_path = init_img_data.replace('.jpg', '.json').replace('.png', '.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    prompts = [json.load(f)['text_input']]
                print("Loaded prompt from json!")

        if prompts is None:
            prompts = []
            for clip_interrogator_mode in clip_interrogator_modes:
                # # Run clip interrogator to get text input:
                prompt = clip_interrogate(StableDiffusionSettings.ckpt, init_image, clip_interrogator_mode, CLIP_INTERROGATOR_MODEL_PATH)
                prompts.append(prompt)

        # Run the actual upscaling:
        for prompt in prompts:
            for checkpoint in checkpoint_options:
                for upscale_init_strength in init_strengths_per_img:
                        # increase the number of pixels if the upscale_init_strength is lower:
                        #target_n_pixels = int(base_target_n_pixels * max(1.0, (1.0 + (0.6 - upscale_init_strength) * 2.5)))
                        target_n_pixels = base_target_n_pixels
                        print(f"\n\nRunning remix with upscale_init_strength: {upscale_init_strength}, target_n_pixels: {target_n_pixels}, ckpt: {os.path.basename(checkpoint)}\nprompt: {prompt}")
                        remix(init_image, prompt, upscale_init_strength, target_n_pixels, steps, img_basename, outdir, checkpoint)

