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

    img_dir = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/controlnet_inputs/neww"
    #img_dir = "/data/xander/Projects/cog/stable-diffusion-dev/eden/xander/init_imgs/test"
    img_paths = [f for f in os.listdir(img_dir) if f.endswith(".jpg") or f.endswith(".png")]
    init_img = os.path.join(img_dir, random.sample(img_paths, 1)[0])

    args = StableDiffusionSettings(
        #ckpt = "stable-diffusion-xl-base-1.0",
        mode = "generate",
        W = random.choice([1024+256]),
        H = random.choice([1024+256]),
        sampler = random.choice(["euler", "euler_ancestral"]),
        steps = 45,
        guidance_scale = random.choice([6,8,10,12]),
        upscale_f = random.choice([1.0, 1.25]),
        text_input = text_input,
        seed = seed,
        n_samples = 1,
        lora_path = None,
        init_image_data = init_img,
        init_image_strength = random.choice([0.45, 0.5, 0.55, 0.6, 0.65]),
        #control_guidance_start = random.choice([0.0, 0.0, 0.05, 0.1]),
        #control_guidance_end = random.choice([0.5, 0.6, 0.7]),
        #control_guidance_end = random.choice([0.65]),
        controlnet_path = "controlnet-luminance-sdxl-1.0",
        #controlnet_path = "controlnet-depth-sdxl-1.0-small",
        #controlnet_path = "controlnet-canny-sdxl-1.0-small",
        #controlnet_path = "controlnet-canny-sdxl-1.0",
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

    
    outdir = "results_controlnet_candy"

    for i in range(200):
        seed = int(time.time())
        #seed = 100+i
        
        seed_everything(seed)

        p2 = list(set(get_prompts_from_json_dir("/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/good_controlnet_jsons")))
        all_p = list(set(text_inputs + sdxl_prompts + p2))
        text_input = random.choice(all_p)

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