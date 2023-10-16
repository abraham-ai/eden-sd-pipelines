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

def download_an_untar(tar_url, output_folder, tar_name):
    """
    Download a tar file from a url, and untar it.
    """
    output_dir = os.path.join(output_folder, tar_name)
    os.makedirs(output_dir, exist_ok = True)
    tar_path = os.path.join(output_folder, tar_name + ".tar")
    if not os.path.exists(tar_path):
        print(f"Downloading {tar_url} to {tar_path}")
        os.system(f"wget {tar_url} -O {tar_path}")

    print(f"Untarring {tar_path}")
    os.system(f"tar -xvf {tar_path} -C {output_dir}")

    print(f"Untarred {tar_path} to {output_dir}")

    return output_dir


modifiers = [
    "trending on artstation",
    "fish eye lens",
    "polaroid photo",
    "poster by ilya kuvshinov katsuhiro", 
    "by magali villeneuve",
    "by jeremy lipkin",
    'by jeremy mann',
    'by jenny saville', 
    'by lucian freud', 
    'by riccardo federici', 
    'by james jean', 
    'by craig mullins', 
    'by jeremy mann', 
    'by makoto shinkai', 
    'by krenz cushart', 
    'by greg rutkowski', 
    'by huang guangjian',
    'by gil elvgren',
    'by lucian poster',
    'by lucian freud',
    'by Conrad roset',
    'by yoshitaka amano',
    'by ruan jia',
    'cgsociety',
]

checkpoint_options = [
    "runwayml:stable-diffusion-v1-5",
    "dreamlike-art:dreamlike-photoreal-2.0",
    "huemin:fxhash_009",
    "eden:eden-v1"
]

# checkpoint_options = ["runwayml:stable-diffusion-v1-5"]
#checkpoint_options = ["eden:eden-v1"]
checkpoint_options = ["stabilityai/stable-diffusion-xl-base-1.0"]

def get_all_img_files(directory_root):
    """
    Recursively get all image files from a directory.
    """
    all_img_paths = []
    for root, dirs, files in os.walk(directory_root):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                all_img_paths.append(os.path.join(root, file))
    return all_img_paths
    

def generate_basic(
    text_input, 
    outdir, 
    steps_per_update = None, # None to disable intermediate frames
    seed = int(time.time()),
    debug = False,
    init_image_data = None,
    lora_path = None,
    prefix = "",
    suffix = ""):

    args = StableDiffusionSettings(
        mode = "generate",
        W = random.choice([1024]),
        H = random.choice([1024]),
        sampler = random.choice(["euler"]),
        steps = 30,
        guidance_scale = random.choice([7]),
        upscale_f = 1.0,
        text_input = text_input,
        seed = seed,
        n_samples = 1,
        init_image_data = init_image_data,
        init_image_strength = 0.0,
        lora_path = lora_path
    )

    name = f'{prefix}{args.text_input[:80]}_{args.seed}_{int(time.time())}{suffix}'
    name = name.replace("/", "_")

    _, imgs = generate(args)

    for i, img in enumerate(imgs):
        save_name = f'{name}_{i}'
        os.makedirs(outdir, exist_ok = True)
        img.save(os.path.join(outdir, save_name + '.jpg'), quality=95)

    # save settings
    settings_filename = f'{outdir}/{name}.json'
    save_settings(args, settings_filename)


if __name__ == "__main__":
    
    outdir = "results_basic"

    for i in range(10):
        seed = int(time.time())
        seed_everything(seed)
        text_input = random.choice(text_inputs)

        generate_basic(i, text_input, outdir, seed = seed)