import sys
sys.path.append('..')

import json
import os
import random
from PIL import Image
import moviepy.editor as mpy

from settings import StableDiffusionSettings
from generation import *
from prompts import text_inputs
from eden_utils import *

def generate_depth2img(init_image_data, text_input, outdir, 
    steps_per_update = None, # None to disable intermediate frames
    seed = int(time.time()),
    debug = False):

    seed_everything(seed)

    args = StableDiffusionSettings(
        mode = "depth2img",
        W = 640,
        H = 640,
        sampler = "euler",
        steps = 50,
        scale = 10,
        upscale_f = 1.0,
        text_input = text_input,
        init_image_strength = 0.15,
        init_image_data = init_image_data,
        seed = seed,
        n_samples = 1,
    )

    # strip all / characters from text_input
    args.text_input = args.text_input.replace("/", "_")
    name = f'{args.text_input[:40]}_{args.seed}_{args.sampler}_{args.steps}'

    generator = make_images(args, steps_per_update=steps_per_update)
    for i, result in enumerate(generator):
        img = result[0]
        for b in range(args.n_samples):
            frame = f"{name}_{b}_{i}.jpg"
            os.makedirs(outdir, exist_ok = True)
            img[b].save(os.path.join(outdir, frame), quality=95)

    # save settings
    settings_filename = f'{outdir}/{name}.json'
    save_settings(args, settings_filename)


if __name__ == "__main__":

    outdir = "results"
    init_image_data = "https://generations.krea.ai/images/3cd0b8a8-34e5-4647-9217-1dc03a886b6a.webp"

    seed = int(time.time())
    seed = 0
    seed_everything(seed)

    text_input = random.choice(text_inputs)
    generate_depth2img(init_image_data, text_input, outdir, seed = seed)