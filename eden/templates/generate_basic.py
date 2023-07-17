import sys
sys.path.append('..')

import json
import os
import random
from PIL import Image
import moviepy.editor as mpy

from settings import StableDiffusionSettings
from generation import *
from prompts import text_inputs, style_modifiers
from eden_utils import *

checkpoint_options = [
    "runwayml:stable-diffusion-v1-5",
    "dreamlike-art:dreamlike-photoreal-2.0",
    "huemin:fxhash_009",
    "eden:eden-v1"
]

# checkpoint_options = ["runwayml:stable-diffusion-v1-5"]
#checkpoint_options = ["eden:eden-v1"]
checkpoint_options = ["stabilityai/stable-diffusion-xl-base-0.9"]

def generate_basic(
    text_input, 
    outdir, 
    steps_per_update = None, # None to disable intermediate frames
    seed = int(time.time()),
    debug = False,
    init_image_data = None,
    prefix = "",
    suffix = ""):

    args = StableDiffusionSettings(
        ckpt = random.choice(checkpoint_options),
        mode = "generate",
        W = 1024,
        H = 1024,
        sampler = "euler",
        steps = 30,
        guidance_scale = 6,
        upscale_f = 1.0,
        text_input = text_input,
        seed = seed,
        n_samples = 1,
        lora_path = None,
        #init_image_data = "/home/xander/Projects/cog/eden-sd-pipelines/eden/assets/huemin_init.jpeg",
        #init_image_strength = 0.15,
    )
    
    name = f'{prefix}{args.text_input[:40]}_{args.seed}_{int(time.time())}{suffix}'

    name = name.replace("/", "_")
    generator = make_images(args)

    for i, img in enumerate(generator):
        frame = f'{name}_{i}.jpg'
        os.makedirs(outdir, exist_ok = True)
        img.save(os.path.join(outdir, frame), quality=95)

    # save settings
    settings_filename = f'{outdir}/{name}.json'
    save_settings(args, settings_filename)


if __name__ == "__main__":

    outdir = "results"
    seed = 0

    seed_everything(seed)
    text_input = "An audio waveform with musicians playing instruments on its peaks and troughs, print design style vector art, bold typography "
    #text_input = random.choice(text_inputs)
    generate_basic(text_input, outdir, seed = seed)