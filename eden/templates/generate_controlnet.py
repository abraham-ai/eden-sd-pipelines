import sys
sys.path.append('..')

import json
import os, shutil
import random
from PIL import Image
import moviepy.editor as mpy

from settings import *
from generation import *
from prompts import *
from eden_utils import *

def generate_basic(
    text_input, 
    outdir, 
    seed = int(time.time()),
    debug = False,
    init_image_data = None,
    prefix = "",
    suffix = ""):

    img_dir = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/poster2_sq"
    ip_dir  = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/ip_images"

    if random.choice([1]):
        ip_img = get_random_imgpath_from_dir(ip_dir)
        text_input = ""
    else:
        ip_img = None

    args = StableDiffusionSettings(
        #mode = "remix",
        #upscale_ckpt = "sdxl-refiner-v1.0",
        W = random.choice([1024+256, 1024+512, 2048]),
        H = random.choice([1024+256]),
        sampler = random.choice(["euler", "euler_ancestral"]),
        steps = 50,
        guidance_scale = random.choice([5,7,9,12]),
        upscale_f = random.choice([1.0, 1.25]),
        text_input = text_input,
        seed = seed,
        n_samples = 1,
        lora_path = None,
        init_image_data = get_random_imgpath_from_dir(img_dir),
        ip_image_data   = ip_img,
        init_image_strength = random.choice([0.35, 0.4, 0.45, 0.5, 0.55]),
        control_guidance_end = random.choice([0.6,0.7,0.8]),
        controlnet_path = "controlnet-luminance-sdxl-1.0", 
        #controlnet_path = random.choice(["controlnet-luminance-sdxl-1.0", "controlnet-luminance-sdxl-1.0", "controlnet-canny-sdxl-1.0-small"]),
        
    )

    controlnet_img = load_img(args.init_image_data, "RGB")

    #name = f'{prefix}{args.text_input[:40]}_{os.path.basename(args.lora_path)}_{args.seed}_{int(time.time())}{suffix}'
    init_img_name = os.path.basename(args.init_image_data).split(".")[0]
    name = f'{prefix}{args.text_input[:40]}_{init_img_name}_{args.seed}_{int(time.time())}{suffix}'
    name = name.replace("/", "_")
    
    generator = make_images(args)

    os.makedirs(outdir, exist_ok = True)
    save_control_img, save_ip_img = True, True

    for i, img in enumerate(generator):
        frame = f'{name}_{i}.jpg'
        img.save(os.path.join(outdir, frame), quality=95)
        if save_control_img:
            controlnet_img.save(os.path.join(outdir, f'{name}_{i}_cond.jpg'), quality=95)
        if save_ip_img and args.ip_image is not None:
            # apply center square crop to the ip_image:
            ip_image = args.ip_image.crop((args.ip_image.width/2 - args.ip_image.height/2, 0, args.ip_image.width/2 + args.ip_image.height/2, args.ip_image.height))
            ip_image.save(os.path.join(outdir, f'{name}_{i}_ip.jpg'), quality=95)
        else:
            Image.new("RGB", (args.W, args.H), (0,0,0)).save(os.path.join(outdir, f'{name}_{i}_ip.jpg'), quality=95)


    # save settings
    settings_filename = f'{outdir}/{name}.json'
    save_settings(args, settings_filename)


if __name__ == "__main__":

    
    outdir = "results_controlnet_poster_ip_sq"
    #outdir = "results_meme_ip_remixes"

    for i in range(2000):
        seed = int(time.time())
        #seed = i
        
        seed_everything(seed)

        p1 = list(set(get_prompts_from_json_dir("/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/good_controlnet_jsons")))
        p2 = list(set(get_prompts_from_json_dir("/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/good_controlnet_jsons2")))
        p3 = list(set(get_prompts_from_json_dir("/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/good_controlnet_jsons3")))
        p4 = list(set(get_prompts_from_json_dir("/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/good_controlnet_jsons_effect")))
        #all_p = list(set(text_inputs + sdxl_prompts + p2))
        text_input = random.choice(p3+p4+p4)


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