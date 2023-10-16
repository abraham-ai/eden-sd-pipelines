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
    debug = 0):

    text_modifiers = [
        "",
        "",
        "",
        "",
        "best quality, sharp details, masterpiece, stunning composition",
        "best quality, sharp details, masterpiece, stunning composition",
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
        clip_interrogator_mode = "fast",
        W = random.choice([1024]),
        H = random.choice([1024]),
        sampler = random.choice(["euler"]),
        steps = 35,
        guidance_scale = random.choice([7]),
        seed = seed,
        n_samples = 1,
        upscale_f = 1.0,
        init_image_strength = random.choice([0.0,0.05]),
        ip_image_strength = random.choice([0.65]),
        init_image_data = init_image_data,
    )

    if debug: # overwrite some args to make things go FAST
        args.W, args.H = 512, 512
        args.steps = 6
        args.n_samples = 1
        args.upscale_f = 1.0

    name = f'remix_{args.init_image_strength:.2f}_{args.ip_image_strength:.2f}_{args.text_input.replace(" ", "_").replace("/", "")}_{args.seed}'

    generator = make_images(args)
    for i, img in enumerate(generator):
        frame = f'{name}_{i}.jpg'
        os.makedirs(outdir, exist_ok = True)
        img.save(os.path.join(outdir, frame), quality=95)

    if 1:
        # Also save the original image:
        init_img = load_img(args.init_image_data, 'RGB').resize((args.W, args.H))
        init_img.save(os.path.join(outdir, f'{name}_original.jpg'), quality=95)

    # save settings
    settings_filename = f'{outdir}/{name}.json'
    save_settings(args, settings_filename)


if __name__ == "__main__":

    outdir = "results_remix"

    init_image_urls = [
        "https://storage.googleapis.com/public-assets-xander/A_workbox/init_imgs/img_00003.jpg",
        "https://storage.googleapis.com/public-assets-xander/A_workbox/init_imgs/img_00005.jpg",
        "https://storage.googleapis.com/public-assets-xander/A_workbox/init_imgs/img_00006.jpg",
        "https://storage.googleapis.com/public-assets-xander/A_workbox/init_imgs/img_00014.jpg",
    ]
    debug = False

    for i in range(10):
        seed = int(time.time())
        init_image_data = random.choice(init_image_urls)
        remix(init_image_data, outdir, seed=seed, debug = debug)

