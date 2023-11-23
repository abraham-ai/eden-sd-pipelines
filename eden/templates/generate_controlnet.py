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

def generate_controlnet(
    text_input, 
    control_img,
    outdir, 
    seed = int(time.time()),
    debug = False,
    init_image = None,
    suffix = ""):

    args = StableDiffusionSettings(
        W = random.choice([1024]),
        H = random.choice([1024]),
        sampler = random.choice(["euler", "euler_ancestral"]),
        steps = 40,
        #use_lcm = True,
        guidance_scale = random.choice([8]),
        upscale_f = random.choice([1.0]),
        text_input = text_input,
        seed = seed,
        n_samples = 1,
        lora_path = None,
        init_image = control_img,
        init_image_strength = random.choice([0.1]),
        control_image   = control_img,
        control_image_strength = random.choice([0.6,0.8,1.0]),
        #ip_image   = ip_img,
        controlnet_path = "controlnet-luminance-sdxl-1.0",
    )

    args.init_image_strength = (1.0 - args.control_image_strength)/random.choice([2.0, 3.0])
    if random.choice([True, False]):
        args.init_image = None

    if args.use_lcm:
        args.steps = int(args.steps / 4)
        addstr = "_LCM"
    else:
        addstr = ""

    name = f'{args.text_input[:40]}_{args.control_image_strength}_{args.controlnet_path}_{args.ckpt}_{args.seed}{addstr}'
    name = name.replace("/", "_")
    os.makedirs(outdir, exist_ok = True)

    save_control_img, save_ip_img = 1,0

    generator = make_images(args)

    for i, img in enumerate(generator):
        frame = f'{name}_{i}.jpg'
        img.save(os.path.join(outdir, frame), quality=95)
        if save_control_img:
            control_img = load_img(args.control_image_path)
            control_img.save(os.path.join(outdir, f'{name}_{i}_cond.jpg'), quality=95)
        if save_ip_img and args.ip_image is not None:
            # apply center square crop to the ip_image:
            ip_image = args.ip_image.crop((args.ip_image.width/2 - args.ip_image.height/2, 0, args.ip_image.width/2 + args.ip_image.height/2, args.ip_image.height))
            ip_image.save(os.path.join(outdir, f'{name}_{i}_ip.jpg'), quality=95)

    # save settings
    settings_filename = f'{outdir}/{name}.json'
    save_settings(args, settings_filename)


if __name__ == "__main__":

    
    outdir = "results_controlnet"

    img_dir = "../assets"
    all_imgs = [f for f in sorted(os.listdir(img_dir)) if f.endswith(".jpg")]
    control_img = os.path.join(img_dir, random.choice(all_imgs))

    for i in range(5):
        seed = int(time.time())
        #seed = i
        
        seed_everything(seed)
        text_input = random.choice(text_inputs)
        generate_controlnet(text_input, control_img, outdir, seed = seed)
