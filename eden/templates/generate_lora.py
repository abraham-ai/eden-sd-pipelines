import sys
sys.path.append('..')

import os

from settings import StableDiffusionSettings
from generation import *
from prompts import text_inputs, style_modifiers
from eden_utils import *

def generate_lora(text_input, outdir, 
    lora_path = None,
    seed = int(time.time()),
    init_image_data = None,
    prefix = "",
    suffix = ""):

    print(text_input)

    args = StableDiffusionSettings(
        lora_path = lora_path,
        lora_scale = 0.8,
        mode = "generate",
        W = 1024,
        H = 1024,
        sampler = "euler",
        steps = 40,
        guidance_scale = 8,
        upscale_f = 1.0,
        text_input = text_input,
        init_image_data = init_image_data,
        seed = seed,
        n_samples = 2,
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
    seed = 1
    lora_path = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/container_lora"
    text_input = "a picture of <concept> drinking coca cola"
    
    seed_everything(seed)
    generate_lora(text_input, outdir, lora_path, seed = seed)