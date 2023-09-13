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
    "runwayml:stable-diffusion-v1-5",
    "dreamlike-art:dreamlike-photoreal-2.0",
    "eden:eden-v1"
]

#checkpoint_options = ["eden:eden-v1"]
checkpoint_options = ["stabilityai/stable-diffusion-xl-base-1.0"]

def generate_basic(
    text_input, 
    outdir, 
    steps_per_update = None, # None to disable intermediate frames
    text_input_2 = None,
    seed = int(time.time()),
    debug = False,
    init_image_data = None,
    prefix = "",
    suffix = ""):

    args = StableDiffusionSettings(
        #ckpt = random.choice(checkpoint_options),
        mode = "generate",
        W = random.choice([1024]),
        H = random.choice([1024]),
        sampler = random.choice(["euler"]),
        steps = 40,
        guidance_scale = random.choice([6,8,10]),
        upscale_f = random.choice([1.0, 1.0]),
        text_input = text_input,
        text_input_2 = text_input_2,
        seed = seed,
        n_samples = 1,
        lora_path = None,
        init_image_data = "/data/xander/Projects/cog/xander_eden_stuff/xander/assets/controlnet/garden/20230803_125517.jpg",
        init_image_strength = random.choice([0.9, 1.0]),
        #controlnet_path = "controlnet-zoe-depth-sdxl-1.0",
        controlnet_path = "controlnet-canny-sdxl-1.0",
        low_t = random.choice([75, 100, 125]),
        high_t = random.choice([150, 200, 250]),
    )

    controlnet_img = load_img(args.init_image_data, "RGB")

    #name = f'{prefix}{args.text_input[:40]}_{os.path.basename(args.lora_path)}_{args.seed}_{int(time.time())}{suffix}'
    name = f'{prefix}{args.text_input[:40]}_{args.seed}_{int(time.time())}{suffix}'
    name = name.replace("/", "_")
    
    generator = make_images(args)

    os.makedirs(outdir, exist_ok = True)
    save_control_img = False

    for i, img in enumerate(generator):
        frame = f'{name}_{i}.jpg'
        img.save(os.path.join(outdir, frame), quality=95)
        #if save_control_img:
        #    control_frame = f'{name}_{i}_cond.jpg'
        #    controlnet_img.save(os.path.join(outdir, control_frame), quality=95)

    # save settings
    settings_filename = f'{outdir}/{name}.json'
    save_settings(args, settings_filename)


if __name__ == "__main__":

    
    outdir = "controlnet_GARDEN"

    text_inputs = [
        "A delicate tapestry of cherry blossom petals",
        ]
        

    for i in range(2):
        seed = random.randint(0, 100000)
        seed = i
        
        seed_everything(seed)
        text_input = random.choice(text_inputs)

        print(text_input)
        if 1:
            generate_basic(text_input, outdir, seed = seed)
        else:
            try:
                generate_basic(text_input, outdir, seed = seed)
            except KeyboardInterrupt:
                print("Interrupted by user")
                exit()  # or sys.exit()
            except Exception as e:
                print(f"Error: {e}")  # Optionally print the error
                continue