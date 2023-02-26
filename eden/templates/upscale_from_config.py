import sys
sys.path.append('..')

import random, os, cv2, time, json, torch, shutil
import numpy as np
from PIL import Image

from settings import StableDiffusionSettings
from generation import *

def compute_target_resolution(W,H,total_pixels):
    # Determine the current target resolution based on total_pixels:
    aspect_ratio = W / H

    # Compute the target resolution:
    W, H = np.sqrt(total_pixels) * np.sqrt(aspect_ratio), np.sqrt(total_pixels) / np.sqrt(aspect_ratio)

    # Round W and H to the nearest multiple of 64:
    W, H = int(np.round(W / 64) * 64), int(np.round(H / 64) * 64)
    return W, H

def adjust_n_steps(steps, init_img_strength, min_max_steps):
    if steps * (1-init_img_strength) > min_max_steps[1]:
        steps = int(min_max_steps[1] / (1-init_img_strength))
    elif steps * (1-init_img_strength) < min_max_steps[0]:
        steps = int(min_max_steps[0] / (1-init_img_strength))
    return min(steps, 999)

def upscale_directory_with_configs(input_cfg_dir, outdir, 
    total_pixels = 1280**2, # higher values will require more GPU memory
    init_img_strengths = [0.65], # How much variation to allow? usually between 0.3 and 0.7, higher = less variation
    n_versions_per_upscale = [1], # How many variations of each image to generate?
    target_steps = 80, min_max_steps = [40,80],
    force_sampler = None, shuffle = False, upscale_lora_scale = 1.0):

    if len(n_versions_per_upscale) != len(init_img_strengths):
        n_versions_per_upscale = [n_versions_per_upscale[0]] * len(init_img_strengths)

    """

    Upscale a directory of SD generated .jpg images (with corresponding .json configs) to a higher resolution.
    Each img file must have the same name as its corresponding .json file.

    This works better than upscale_unconditioned.py, but requires a config file for each image.

    """

    # Keys to ignore when loading config:
    ignore_keys = ['steps', 'init_image_file', 'init_image_strength', 'n_samples', 'init_image', 'init_latent', 'init_image_b64', 'init_sample', 'init_image_inpaint_mode']


    if 0:
        # Get all img files and configs:
        cfg_files = [os.path.splitext(f)[0] for f in os.listdir(input_cfg_dir) if ".json" in f]
        img_files = [os.path.splitext(f)[0]  for f in os.listdir(input_cfg_dir) if ".jpg"  in f]

        # Get intersection and add back the extensions:
        filenames = sorted(list(set(cfg_files) & set(img_files)))
        print(f"Found {len(filenames)} files to upscale.")
        
        if shuffle:
            seed_everything(int(time.time()))
            random.shuffle(filenames)

        cfg_files = [os.path.join(input_cfg_dir, f + '.json') for f in filenames]
        img_files = [os.path.join(input_cfg_dir, f + '.jpg')  for f in filenames]
    
    else:
        # Get all img files and configs:
        cfg_files = sorted([os.path.join(input_cfg_dir, f) for f in os.listdir(input_cfg_dir) if ".json" in f])
        img_files = sorted([os.path.join(input_cfg_dir, f)  for f in os.listdir(input_cfg_dir) if ".jpg"  in f])
        print(f"Found {len(img_files)} files to upscale.")


    args = StableDiffusionSettings(
        #ckpt   = ckpt,
        steps = target_steps,
    )
    
    for i in range(len(img_files)):
        for j, init_img_strength in enumerate(init_img_strengths):
            for jj in range(n_versions_per_upscale[j]):

                print(f"Upscaling {os.path.basename(cfg_files[i])}..")
                
                with open(cfg_files[i]) as json_file:
                    cfg_data = json.load(json_file)
                
                # Overwrite all attributes in args with the keys in cfg_data:
                for key in cfg_data:
                    if key not in ignore_keys:
                        setattr(args, key, cfg_data[key])

                args.lora_scale = upscale_lora_scale
                args.lora_path = "/home/xander/Projects/cog/lora/exps/kirby/kirby_train_00_563e0a/final_lora.safetensors"
                args.guidance_scale = random.choice([12])

                if force_sampler is not None:
                    args.sampler = force_sampler


                args.W, args.H = compute_target_resolution(args.W, args.H, total_pixels)
                args.init_image_data = img_files[i]
                args.init_image_strength = init_img_strength
                args.seed = int(time.time())
                args.steps = adjust_n_steps(args.steps, init_img_strength, min_max_steps)

                filename = f"{os.path.basename(img_files[i])}_HD_{init_img_strength:.2f}_{args.guidance_scale}_{jj:02d}"
                outfilepath = os.path.join(outdir, filename+".jpg")

                if os.path.exists(os.path.join(outdir, filename+".json")):
                    print(f"Skipping {filename}..")
                    continue

                if not args.init_image_data.endswith(".jpg"):
                    continue

                _, new_images = generate(args)
                frame = new_images[0]
                os.makedirs(outdir, exist_ok = True)
                frame.save(outfilepath, quality=95)
                
                if 1: # save config json to disk:
                    with open(os.path.join(outdir, filename+".json"), 'w') as fp:
                        json.dump(vars(args), fp, default=lambda o: '<not serializable>', indent=2)

"""


cd /home/xander/Projects/cog/eden-sd-pipelines/eden/templates
python upscale_from_config.py



"""


input_cfg_dir  = "/home/xander/Projects/cog/eden-sd-pipelines/eden/xander/images/kirby/good/more2"
outdir         = input_cfg_dir + "_HD"

##################################################################################################

if __name__ == "__main__":
    upscale_lora_scale = 0.9

    if 0:  # make variations:
        upscale_directory_with_configs(input_cfg_dir, outdir, 
            total_pixels = (1024+246)**2, 
            init_img_strengths = [0.25, 0.30, 0.35, 0.40], 
            n_versions_per_upscale = [1,2,2,2],
            force_sampler = 'euler_ancestral',
            shuffle = True,
            target_steps = 100, min_max_steps = [40,70],
            upscale_lora_scale = upscale_lora_scale,
            )

    else: # upscale:

        upscale_directory_with_configs(input_cfg_dir, outdir, 
            total_pixels = (1024)**2, 
            init_img_strengths = [0.3,0.35,0.4,0.45,0.50], 
            n_versions_per_upscale = [1],
            #force_sampler = 'euler_ancestral',
            shuffle = True,
            target_steps = 160, min_max_steps = [140,200],
            upscale_lora_scale = upscale_lora_scale,
            )