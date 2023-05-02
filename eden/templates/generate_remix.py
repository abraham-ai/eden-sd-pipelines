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

    args = StableDiffusionSettings(
        mode = "remix",
        clip_interrogator_mode = "fast",
        W = 768,
        H = 768,
        sampler = "euler",
        steps = 30,
        guidance_scale = 8,
        seed = seed,
        n_samples = 2,
        upscale_f = 1.5,
        init_image_strength = 0.175,
        init_image_data = init_image_data,
        #aesthetic_steps = 15,
        #aesthetic_lr = 0.0001,
        #ag_L2_normalization_constant = 0.05
    )

    if debug: # overwrite some args to make things go FAST
        args.W, args.H = 512, 512
        args.steps = 25
        args.n_samples = 2
        args.upscale_f = 1.1

    name = f'remix_{args.seed}_{args.sampler}_{args.steps}_{int(time.time())}'

    generator = make_images(args)
    for i, img in enumerate(generator):
        frame = f'{name}_{i}.jpg'
        os.makedirs(outdir, exist_ok = True)
        img.save(os.path.join(outdir, frame), quality=95)

    # Also save the original image:
    args.init_image.save(os.path.join(outdir, f'remix_original.jpg'), quality=95)

    # save settings
    settings_filename = f'{outdir}/{name}.json'
    save_settings(args, settings_filename)


if __name__ == "__main__":

    outdir = "results"
    init_image_data = "https://generations.krea.ai/images/3cd0b8a8-34e5-4647-9217-1dc03a886b6a.webp"
    seed = int(time.time())
    seed = 0
    remix(init_image_data, outdir, seed=seed)

