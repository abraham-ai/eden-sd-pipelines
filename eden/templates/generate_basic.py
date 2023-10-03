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
    text_input_2 = None,
    seed = int(time.time()),
    debug = False,
    init_image_data = None,
    lora_path = None,
    prefix = "",
    suffix = "_freeU_mild3"):


    init_imgs = [None]
    init_img_dir = "/data/xander/Projects/cog/stable-diffusion-dev/eden/xander/img2img_inits"
    init_img_path = random.choice(get_all_img_files(init_img_dir))

    args = StableDiffusionSettings(
        mode = "generate",
        W = random.choice([1024]),
        H = random.choice([1024]),
        sampler = random.choice(["euler"]),
        steps = 30,
        guidance_scale = random.choice([8]),
        upscale_f = 1.0,
        text_input = text_input,
        text_input_2 = text_input_2,
        seed = seed,
        n_samples = 1,
        #init_image_data = random.choice([None, None, init_img_path]),
        #init_image_strength = random.choice([0.05, 0.1, 0.15, 0.2, 0.25]),
        #lora_path = "/data/xander/Projects/cog/GitHub_repos/cog-sdxl/lora_models_saved/koji_color/checkpoints/checkpoint-804"
    )

    name = f'{prefix}{args.text_input[:80]}_{args.seed}_{int(time.time())}{suffix}'
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
    
    outdir = "results_freeU"
    
    # remove the output directory
    if os.path.exists(outdir) and 0:
        os.system(f"rm -rf {outdir}")

    for i in range(20):
        seed = random.randint(0, 100000)
        seed = i

        seed_everything(seed)
        text_input = random.choice(text_inputs)

        #text_input = "a beautiful mountain landscape"

        text_input = random.choice([
            "a photo of Nones",
        ])

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