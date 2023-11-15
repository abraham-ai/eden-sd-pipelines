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
        text_input = "",
        clip_interrogator_mode = "fast",
        W = random.choice([2048]),
        H = random.choice([1024]),
        sampler = random.choice(["euler", "euler_ancestral"]),
        steps = 50,
        guidance_scale = random.choice([8]),
        adopt_aspect_from_init_img = True,
        #use_lcm = True,
        noise_sigma = 0.2, #0.25
        seed = seed,
        n_samples = 1,
        upscale_f = 1.0,
        control_image = init_image,
        controlnet_path = random.choice(["controlnet-canny-sdxl-1.0-small", "controlnet-luminance-sdxl-1.0"]),
        control_image_strength = random.choice([0.6, 0.7, 0.8]),
        ip_image_strength = random.choice([0.6]),
        init_image_strength = random.choice([0.0, 0.05, 0.1, 0.65, 0.7,]),
        init_image = init_image,
    )

    if args.use_lcm:
        args.steps = int(args.steps / 4)

    #args.H = int(args.W * 3 / 4)

    if debug: # overwrite some args to make things go FAST
        args.W, args.H = 512, 512
        args.steps = 6
        args.n_samples = 1
        args.upscale_f = 1.0

    lcm_string = "_lcm" if args.use_lcm else ""

    prompt_name = args.text_input[:60].replace(" ", "_").replace("/", "")
    name = f'remix{lcm_string}_{args.init_image_strength:.2f}_{args.ip_image_strength:.2f}_{prompt_name}_{args.seed}'
    generator = make_images(args)

    for i, img in enumerate(generator):
        frame = f'{name}_{i}.jpg'
        os.makedirs(outdir, exist_ok = True)
        img.save(os.path.join(outdir, frame), quality=95)

    if 0:
        # Also save the original image:
        init_img = load_img(init_image, 'RGB').resize((args.W, args.H))
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

    init_image_urls = ["/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/poster/WhatsApp Image 2023-10-21 at 15.30.55.jpeg", "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/destelbergen/to_remix/1ac0fbdf-45bf-48fc-83e2-1283d48d9282.jpeg"]

    debug = False

    for i in range(40):
        seed = int(time.time())
        init_image = random.choice(init_image_urls)
        remix(init_image, outdir, seed=seed, debug = debug)

