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
    "runwayml/stable-diffusion-v1-5",
    "prompthero/openjourney-v2",
    "dreamlike-art/dreamlike-photoreal-2.0"
]
checkpoint_options = ["dreamlike-art/dreamlike-photoreal-2.0"]

def generate_basic(text_input, outdir, 
    steps_per_update = None, # None to disable intermediate frames
    seed = int(time.time()),
    debug = False,
    init_image_data = None,
    prefix = "",
    suffix = ""):

    print(text_input)

    args = StableDiffusionSettings(
        ckpt = random.choice(checkpoint_options),
        mode = "generate",
        W = 512,
        H = 512,
        sampler = "euler",
        steps = 40,
        scale = 12,
        upscale_f = 1.0,
        text_input = text_input,
        seed = seed,
        n_samples = 1,
        lora_path = None,
        #init_image_data = init_image_data,
        #init_image_strength = 0.25,
    )
    
    name = f'{prefix}{args.text_input[:40]}_{args.seed}_{int(time.time())}{suffix}'

    name = name.replace("/", "_")
    generator = make_images(args, steps_per_update=steps_per_update)

    for i, img in enumerate(generator):
        frame = f'{name}_{i}.jpg'
        os.makedirs(outdir, exist_ok = True)
        img.save(os.path.join(outdir, frame), quality=95)

    # save settings
    settings_filename = f'{outdir}/{name}.json'
    save_settings(args, settings_filename)


if __name__ == "__main__":

    outdir = "results"
    seed = 1

    seed_everything(seed)
    text_input = random.choice(text_inputs)
    generate_basic(text_input, outdir, seed = seed)