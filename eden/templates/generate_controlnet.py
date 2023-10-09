import sys
sys.path.append('..')

import json
import os
import random
from PIL import Image
import moviepy.editor as mpy

from settings import *
from generation import *
from prompts import *
from eden_utils import *


checkpoint_options = [
    "runwayml:stable-diffusion-v1-5",
    "dreamlike-art:dreamlike-photoreal-2.0",
    "eden:eden-v1"
]

def generate_basic(
    text_input, 
    outdir, 
    seed = int(time.time()),
    debug = False,
    init_image_data = None,
    prefix = "",
    suffix = ""):

    img_dir = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/poster"
    #img_dir = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/memes"

    img_paths = [f for f in os.listdir(img_dir) if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg")]
    init_img = os.path.join(img_dir, random.sample(img_paths, 1)[0])

    args = StableDiffusionSettings(
        #mode = "remix",
        W = random.choice([1024+256, 1024+512]),
        H = random.choice([1024+256]),
        sampler = random.choice(["euler", "euler_ancestral"]),
        steps = 50,
        guidance_scale = random.choice([5,7,9]),
        upscale_f = random.choice([1.3]),
        text_input = text_input,
        seed = seed,
        n_samples = 1,
        lora_path = None,
        init_image_data = init_img,
        init_image_strength = random.choice([0.4, 0.45, 0.5, 0.55, 0.6]),
        #control_guidance_start = random.choice([0.0, 0.0, 0.05, 0.1]),
        #control_guidance_end = random.choice([0.5, 0.6, 0.7]),
        #control_guidance_end = random.choice([0.65]),
        #controlnet_path = "controlnet-luminance-sdxl-1.0", 
        #controlnet_path = "controlnet-depth-sdxl-1.0-small",
        #controlnet_path = "controlnet-canny-sdxl-1.0-small",
        #controlnet_path = "controlnet-canny-sdxl-1.0",
        controlnet_path = random.choice(["controlnet-luminance-sdxl-1.0", "controlnet-luminance-sdxl-1.0", "controlnet-luminance-sdxl-1.0", "controlnet-luminance-sdxl-1.0", "controlnet-canny-sdxl-1.0", "controlnet-depth-sdxl-1.0-small"]),
    )

    controlnet_img = load_img(args.init_image_data, "RGB")

    #name = f'{prefix}{args.text_input[:40]}_{os.path.basename(args.lora_path)}_{args.seed}_{int(time.time())}{suffix}'
    name = f'{prefix}{args.text_input[:40]}_{args.seed}_{int(time.time())}{suffix}'
    name = name.replace("/", "_")
    
    generator = make_images(args)

    os.makedirs(outdir, exist_ok = True)
    save_control_img = True

    for i, img in enumerate(generator):
        frame = f'{name}_{i}.jpg'
        img.save(os.path.join(outdir, frame), quality=95)
        if save_control_img:
            control_frame = f'{name}_{i}_cond.jpg'
            controlnet_img.save(os.path.join(outdir, control_frame), quality=95)

    # save settings
    settings_filename = f'{outdir}/{name}.json'
    save_settings(args, settings_filename)


if __name__ == "__main__":

    
    outdir = "results_controlnet_poster_01"
    #outdir = "results_meme_ip_remixes"

    for i in range(2000):
        seed = int(time.time())
        #seed = i
        
        seed_everything(seed)

        p1 = list(set(get_prompts_from_json_dir("/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/good_controlnet_jsons")))
        p2 = list(set(get_prompts_from_json_dir("/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/good_controlnet_jsons2")))
        p3 = list(set(get_prompts_from_json_dir("/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/good_controlnet_jsons3")))
        #all_p = list(set(text_inputs + sdxl_prompts + p2))
        text_input = random.choice(p1+p2+p3)


        if 1:
            n_style_modifiers = random.choice([0,0,1,2])
            if n_style_modifiers > 0:
                text_input = text_input + ", " + ", ".join(random.sample(style_modifiers, n_style_modifiers))

        print(text_input)
        if 0:
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