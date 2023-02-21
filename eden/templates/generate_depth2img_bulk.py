import sys
sys.path.append('..')

import json
import os
import random
from PIL import Image
import moviepy.editor as mpy

from settings import StableDiffusionSettings
from generation import *
from prompts import text_inputs
from eden_utils import *

def generate_depth2img(init_image_data, text_input, outdir, 
    steps_per_update = None, # None to disable intermediate frames
    seed = int(time.time()),
    init_strength = 0.15,
    debug = False):

    seed_everything(seed)

    args = StableDiffusionSettings(
        mode = "depth2img",
        W = 576,
        H = 576,
        sampler = "euler",
        steps = 40,
        guidance_scale = 10,
        upscale_f = 1.0,
        text_input = text_input,
        init_image_strength = init_strength,
        init_image_data = init_image_data,
        seed = seed,
        n_samples = 1,
    )

    # strip all / characters from text_input
    args.text_input = args.text_input.replace("/", "_")
    name = f'{args.text_input[:40]}_{args.seed}_{args.sampler}_{args.steps}'

    generator = make_images(args)
    for i, result in enumerate(generator):
        img = result[0]
        for b in range(args.n_samples):
            frame = f"{name}_{b}_{i}.jpg"
            os.makedirs(outdir, exist_ok = True)
            img[b].save(os.path.join(outdir, frame), quality=95)

    # save settings
    settings_filename = f'{outdir}/{name}.json'
    save_settings(args, settings_filename)


if __name__ == "__main__":

    """

    export CUDA_VISIBLE_DEVICES=1
    cd /home/xander/Projects/cog/diffusers/eden/templates
    python3 generate_depth2img_bulk.py

    """

    outdir = "results"
    input_img_dir = "/home/xander/Pictures/Mars2023/vince"

    text_inputs = [
        "a sympathetic caveman, character portrait by greg rutkowski, craig mullins",
        "an accidental selfies taken by a caveman in 10000 bc, Canon EOS R3, f/1.4, ISO 200, 1/160s, 8K, RAW, unedited, symmetrical balance, in-frame",
        "a sympathetic caveman, character portrait by greg rutkowski, craig mullins ",
        "hairy beast with club, swamp, richard kane - ferguson ",
        "A photo of king conan the barbarian sitting on his throne, award winning photography, sigma 85mm Lens F/1.4, blurred background, perfect faces",
        "portrait of a forest hermit wearing a driftwood mask in an ominous forest, photography ",
        "photorealistic photograph of bigfoot, 3 5 mm film, fuji, leica s, bigfoot, nyc, in the style of fashion photography, intricate, golden hour ",
        "National Geographic photo of yowie in the Australian bush",
        "Cookie Monster Muppet on Sesame Street eating pizza in secret, happy",
        "Cookie Monster Muppet on Sesame Street smoking weed, happy",
        "sumo wrestler baby ",
        "Sumo wrestler smoking a cigarette, high quality photo",
        "Large tardigrade with fuzzy!!!!! fur!!!!!, trending on artstation, photorealistic imagery, heavily detailed, intricate, 4k, 8k, artstation graphics, artstation 3d render, artstation 3d, artstation unreal engine",
        "two long haired capybara",
        "big sir monster is a hybrid of shrek, big foot, elephant, and hippo",
        "Very very very very highly detailed epic central composition photo of Mr Bean as Shrek face, intricate, extremely detailed, digital painting, smooth, sharp focus, illustration, happy lighting, incredible art by Brooke Shaden, artstation, concept art, Octane render in Maya and Houdini",
        "shrek in a milk pool, 4 k, art station concept, highly detailed",
        "kevin james as shrek, movie still",
    ]

    text_inputs = [
        "a sympathetic caveman, wearing a black coat, character portrait by greg rutkowski, craig mullins",
        "an accidental selfie taken by a caveman in 10000 bc, Canon EOS R3, f/1.4, ISO 200, 1/160s, 8K, RAW, unedited, symmetrical balance, in-frame",
        "a sympathetic caveman, character portrait by greg rutkowski, craig mullins ",
        "hairy beast with club, swamp, richard kane - ferguson ",
        "A photo of king conan the barbarian sitting on his throne, award winning photography, sigma 85mm Lens F/1.4, blurred background, perfect faces",
        "portrait of a forest hermit wearing a driftwood mask in an ominous forest, photography ",
        "photorealistic photograph of bigfoot, 3 5 mm film, fuji, leica s, bigfoot, nyc, in the style of fashion photography, intricate, golden hour ",
        "National Geographic photo of yowie in the Australian bush",
        "Cookie Monster Muppet wearing a black coat, on Sesame Street eating pizza in secret, happy",
        "Cookie Monster Muppet on Sesame Street smoking weed, happy",
        "sumo wrestler baby",
        "Sumo wrestler smoking a cigarette, high quality photo",
        "Large tardigrade with fuzzy!!!!! fur!!!!!, trending on artstation, photorealistic imagery, heavily detailed, intricate, 4k, 8k, artstation graphics, artstation 3d render, artstation 3d, artstation unreal engine",
        "two long haired capybara",
        "big sir monster is a hybrid of shrek, big foot, elephant, and hippo",
        "Very very very very highly detailed epic central composition photo of Mr Bean as Shrek face, intricate, extremely detailed, digital painting, smooth, sharp focus, illustration, happy lighting, incredible art by Brooke Shaden, artstation, concept art, Octane render in Maya and Houdini",
        "shrek in a milk pool, 4 k, art station concept, highly detailed",
        "kevin james as shrek, movie still",
    ]

    for i in range(100):
        seed = int(time.time())
        seed_everything(seed)

        init_image_data = os.path.join(input_img_dir, random.choice(os.listdir(input_img_dir)))
        init_strength = random.uniform(0.35, 0.65)
        
        text_input = random.choice(text_inputs)
        generate_depth2img(init_image_data, text_input, outdir, init_strength = init_strength, seed = seed)