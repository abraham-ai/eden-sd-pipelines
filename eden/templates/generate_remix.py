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
from prompts import text_inputs
from eden_utils import *


def remix(init_image_data, outdir, 
    steps_per_update = None, # None to disable intermediate frames
    seed = int(time.time()),
    debug = False):

    text_modifiers = [
        "",
        "",
        "",
        "",
        "tilt shift photo, macrophotography",
        "pixel art, 16-bit, pixelated",
        "cubism, abstract art",
        "on the beach",
        "butterfly, 🦋",
        "low poly, geometric shapes",
        "origami, paper folds",
        "drawing by M.C. Escher",
        "painting by Salvador Dalí",
        "painting by Wassily Kandinsky",
        "H. R. Giger, biomechanical",
        "starry night, Van Gogh swirls",
    ]

    args = StableDiffusionSettings(
        mode = "remix",
        text_input = random.choice(text_modifiers),
        ip_image_strength = random.choice([0.4,0.45,0.5,0.55,0.6]),
        clip_interrogator_mode = "fast",
        W = random.choice([1024, 1024+256]),
        H = random.choice([1024, 1024+256]),
        sampler = random.choice(["euler", "euler_ancestral"]),
        steps = 50,
        guidance_scale = random.choice([6,8,10]),
        seed = seed,
        n_samples = 2,
        upscale_f = 1.25,
        #init_image_strength = 0.0,
        init_image_strength = random.choice([0.0,0.05]),
        init_image_data = init_image_data,
    )

    if debug: # overwrite some args to make things go FAST
        args.W, args.H = 512, 512
        args.steps = 25
        args.n_samples = 2
        args.upscale_f = 1.1

    name = f'remix_{args.seed}_{int(time.time())}_{args.ip_image_strength}_{args.text_input.replace(" ", "_")}'
    name = f'{args.init_image_strength:.2f}_{args.ip_image_strength:.2f}_{args.text_input.replace(" ", "_")}_{args.seed}'

    generator = make_images(args)
    for i, img in enumerate(generator):
        frame = f'{name}_{i}.jpg'
        os.makedirs(outdir, exist_ok = True)
        img.save(os.path.join(outdir, frame), quality=95)

    # Also save the original image:
    init_img = load_img(args.init_image_data, 'RGB').resize((args.W, args.H))
    init_img.save(os.path.join(outdir, f'{name}_original.jpg'), quality=95)

    # save settings
    settings_filename = f'{outdir}/{name}.json'
    save_settings(args, settings_filename)


if __name__ == "__main__":

    outdir = "results_remix_fin_prn"
    init_image_data = "https://generations.krea.ai/images/3cd0b8a8-34e5-4647-9217-1dc03a886b6a.webp"


    for i in range(200):
        seed = int(time.time())
        input_dir = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/01_great_inits"
        init_image_data = os.path.join(input_dir, random.choice(os.listdir(input_dir)))

        remix(init_image_data, outdir, seed=seed)

