import sys
sys.path.append('..')

import os
from settings import StableDiffusionSettings
from generation import *
from prompts import text_inputs, style_modifiers, lora_prompts
from eden_utils import *

def generate_lora(text_input, outdir, 
    lora_path = None,
    seed = int(time.time()),
    init_image = None,
    prefix = "",
    suffix = ""):

    print(text_input)

    args = StableDiffusionSettings(
        lora_path = lora_path,
        lora_scale = random.choice([0.6, 0.7, 0.8, 0.9]),
        mode = "generate",
        W = random.choice([1024]),
        H = random.choice([1024]),
        steps = 35,
        noise_sigma = 0.0,
        guidance_scale = random.choice([8,10]),
        upscale_f = 1.0,
        text_input = text_input,
        init_image = init_image,
        seed = seed,
        n_samples = 1,
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

    outdir = "results_lora_xander"
    lora_paths = [
        "/data/xander/Projects/cog/GitHub_repos/cog-sdxl/lora_models/objects_banny_best.zip_002_4959/checkpoints/checkpoint-1200"
    ]

    prompt_file = "../random_prompts.txt"
    text_inputs = open(prompt_file).read().split("\n")
    text_inputs = lora_prompts

    for i in range(200):
        seed = int(time.time())
        seed_everything(seed)
        text_input = random.choice(text_inputs)
        lora_path  = random.choice(lora_paths)
        try:
            generate_lora(text_input, outdir, lora_path, seed = seed)
        except Exception as e:
            print(str(e))
            time.sleep(0.5)
            continue