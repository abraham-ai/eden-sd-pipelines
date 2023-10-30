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


def remix(init_image, outdir, 
    steps_per_update = None, # None to disable intermediate frames
    seed = int(time.time()),
    debug = 0):

    text_modifiers = [
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
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
        W = random.choice([1024+512, 2048]),
        H = random.choice([1024+512]),
        sampler = random.choice(["euler", "euler_ancestral"]),
        steps = 60,
        guidance_scale = random.choice([4,6,7,8]),
        adopt_aspect_from_init_img = True,
        seed = seed,
        n_samples = 1,
        upscale_f = 1.0,
        #init_image_strength = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        init_image_strength = random.choice([0.8, 0.85, 0.9]),
        ip_image_strength = random.choice([0.4,0.6,0.8]),
        init_image = init_image,
    )

    #args.H = int(args.W * 3 / 4)

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

    if 0:
        # Also save the original image:
        init_img = load_img(args.init_image, 'RGB').resize((args.W, args.H))
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

    for i in range(20):
        seed = int(time.time())
        init_image = random.choice(init_image_urls)

        init_image = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/poster/WhatsApp Image 2023-10-21 at 15.30.55.jpeg"
        init_image = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/destelbergen/to_remix/1ac0fbdf-45bf-48fc-83e2-1283d48d9282.jpeg"
        remix(init_image, outdir, seed=seed, debug = debug)

