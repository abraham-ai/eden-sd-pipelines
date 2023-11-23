import sys
sys.path.append('..')

import json
import time
import os
import random
from PIL import Image
import moviepy.editor as mpy

from settings import StableDiffusionSettings
from generation import *
from prompts import text_inputs, style_modifiers
from eden_utils import *

def generate_basic(
    text_input, 
    outdir, 
    steps_per_update = None, # None to disable intermediate frames
    seed = int(time.time()),
    debug = False,
    init_image = None,
    lora_path = None,
    prefix = "",
    suffix = "",
    iteration = 0):

    args = StableDiffusionSettings(
        mode = "generate",
        #use_lcm = True,
        W = random.choice([1024]),
        H = random.choice([1024]),
        sampler = random.choice(["euler"]),
        steps = 30,
        guidance_scale = 8,
        text_input = text_input,
        seed = seed,
        n_samples = 1,
        #init_image = init_image,
        #init_image_strength = 0.0,
    )

    if args.use_lcm:
        args.steps = int(args.steps / 4)
        args.guidance_scale = 0.0
        addstr = f"_LCM_{args.steps}_steps"
    else:
        addstr = f"_no_LCM_base_{args.steps}_steps"

    name = f'{args.text_input[:60]}{addstr}_{args.guidance_scale}_{args.seed}_{args.ckpt}_{int(time.time())}'
    name = name.replace("/", "_")
    start_time = time.time()

    _, imgs = generate(args)

    time_delta = time.time() - start_time
    print(f"Generated in {time_delta:.2f} seconds")

    for i, img in enumerate(imgs):
        save_name = f'{name}_{i}'
        os.makedirs(outdir, exist_ok = True)
        filepath = os.path.join(outdir, save_name + '.jpg')
        img.save(filepath, quality=95)

    # save settings
    settings_filename = f'{outdir}/{name}.json'
    save_settings(args, settings_filename)


if __name__ == "__main__":
    
    outdir = "results_basic"

    for i in range(5):
        seed = int(time.time())
        #seed = i
        seed_everything(seed)
        text_input = random.choice(text_inputs)
        generate_basic(text_input, outdir, seed = seed, iteration = i)