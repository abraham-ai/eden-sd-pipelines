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
    "huemin/fxhash_009"
]

def generate_basic(text_input, outdir, 
    steps_per_update = None, # None to disable intermediate frames
    seed = int(time.time()),
    debug = False,
    init_image_data = "../assets/huemin_init.jpeg",
    prefix = "",
    suffix = ""):

    print(text_input)

    args = StableDiffusionSettings(
        ckpt = random.choice(checkpoint_options),
        mode = "generate",
        W = 768,
        H = 768,
        sampler = "euler",
        steps = 100,
        guidance_scale = 8,
        upscale_f = 1.0,
        text_input = text_input,
        uc_text = "mosaic maze 2d cloth texture simple rough saturated detailed green grid",
        seed = seed,
        n_samples = 1,
        lora_path = None,
        init_image_data = init_image_data,
        init_image_strength = 0.2,
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
    seed = 2185023741

    seed_everything(seed)
    text_input = "isometric swirl made of solid rectangles, a screenprint, pastel oil on canvas, hd wallpaper, sculpture curves, 3d outline shader render isometric"
    generate_basic(text_input, outdir, seed = seed)