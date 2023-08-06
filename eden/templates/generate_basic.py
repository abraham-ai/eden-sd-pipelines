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
    "huemin:fxhash_009",
    "eden:eden-v1"
]

# checkpoint_options = ["runwayml:stable-diffusion-v1-5"]
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
    suffix = "sdxl1.0"):

    init_img = random.choice([
        "/data/xander/Projects/cog/xander_eden_stuff/xander/assets/hetty/wimesteban_a_middle-aged_short_haired_blonde_female_cyborg_prof_0db6473d-2545-47d7-97c3-18eeae9f00da_ins (1).jpg",
        "/data/xander/Projects/cog/xander_eden_stuff/xander/assets/hetty/enlarge_wimesteban_a_middle-aged_blonde_female_cyborg_professor_with_ar_ef917cc9-778e-4c30-9d35-977c68bcd051_ins.jpg",
        "/data/xander/Projects/cog/xander_eden_stuff/xander/assets/hetty/enlarge_wimesteban_a_middle-aged_blonde_female_cyborg_professor_with_ar_ef917cc9-778e-4c30-9d35-977c68bcd051_ins.jpg",
        "/data/xander/Projects/cog/xander_eden_stuff/xander/assets/hetty/enlarge_wimesteban_a_middle-aged_blonde_female_cyborg_professor_with_ar_ef917cc9-778e-4c30-9d35-977c68bcd051_ins.jpg",
        None
    ])

    args = StableDiffusionSettings(
        #ckpt = random.choice(checkpoint_options),
        mode = "generate",
        W = 2300,
        H = 2300,
        sampler = "euler",
        steps = 80,
        guidance_scale = random.choice([5,7,9]),
        upscale_f = 1.0,
        text_input = text_input,
        text_input_2 = text_input_2,
        seed = seed,
        n_samples = 1,
        #lora_path = None,
        lora_path = "/data/xander/Projects/cog/diffusers/lora/trained_models/sdxl-lora-hetty-reg/checkpoint-400",
        init_image_data = init_img,
        init_image_strength = random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
    )

    if 0: #plantoid:
        args = StableDiffusionSettings(
            mode = "generate",
            W = random.choice([1024, 1024+256, 1024+512]),
            H = random.choice([1024, 1024+256, 1024+512]),
            sampler = "euler",
            steps = 60,
            guidance_scale = random.choice([5,7,9]),
            upscale_f = 1.0,
            text_input = text_input,
            text_input_2 = text_input_2,
            seed = seed,
            n_samples = 1,
            lora_path = "/data/xander/Projects/cog/diffusers/lora/trained_models/sdxl-lora-plantoid",
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

    if 0:
        outdir = "results_plantoid2"

        text_inputs = [
            "a photo of a sks plantoid in a beautiful portugese garden made from a boat anchor chain and an engine gear as head, scrap metal, welded",
            "a photo of a sks plantoid in a beautiful garden made from an anchor chain and an engine gear head, scrap metal",
            "a photo of a sks plantoid on an iceberg in the middle of the ocean, moonlit",
            "a photo of two sks plantoids entangled and in love, masterpiece artwork, glowing hearts",
            "a magnificent sks plantoid in the desert",
            "a graceful sks plantoid standing in the middle of a courtyard, surrounded by flowers",
            "a magnificent sks plantoid standing in the middle of a medievel walled courtyard, surrounded by flowers",
            "a magnificent, huge sks plantoid standing in the middle of a huge museum made of glass, surrounded by people",
        ]

        text_inputs = [
            "a photo of a massive sks plantoid towering high above the New York skyline, apocalypse, sks plantoid taking over the world",
            "a strong, armored sks plantoid fighting a polar bear",
            "a sks plantoid sitting on the Iron Throne, game of thrones",
            "a photo of a sks plantoid in the turret of a tank, going into war",
            "a photo of a sks plantoid in a cage fight, surrounded by the crowd",
            "a sks plantoid taking acid, LSD visual",
        ]

    else:
        outdir = "results_hetty"
        text_inputs = [
            "Envision a beautiful dauntless sks woman, her physique shielded by an avant-garde suit, 2019 sci-fi masterpiece. Hyper-modern tunnel, reminiscent of an 8K snapshot from 'Prometheus'. Amidst the surreal cosmos, highly detailed VFX-rendered spacesuit, exuding both allure and power., poster by ilya kuvshinov katsuhiro, by magali villeneuve",
            "Beautiful sks woman with short blond hair standing in the corridor of an intergalactic spaceship, wearing spacesuit, helmet under the arm, depth and fuzzy lighting in the background, by ruan jia, trending on artstation, fish eye lens, by jeremy mann",
            "A beautiful middle aged blonde female cyborg sks woman with army decoration in a white spaceship, wearing super detail futuristic chrome silver armor full body in a big wide background, clear facial features, cinematic, in the style of blade runner and alan moore, 35mm lens, accent lighting, global illumination, polaroid photo",
            "Envision a beautiful dauntless sks woman, her physique shielded by an avant-garde suit, 2019 sci-fi masterpiece. Hyper-modern tunnel, reminiscent of an 8K snapshot from 'Prometheus'. Amidst the surreal cosmos, highly detailed VFX-rendered spacesuit, exuding both allure and power., poster by ilya kuvshinov katsuhiro, by magali villeneuve",
            "Beautiful sks woman with short blond hair standing in the corridor of an intergalactic spaceship, wearing spacesuit, helmet under the arm, depth and fuzzy lighting in the background, by ruan jia, trending on artstation, fish eye lens, by jeremy mann",
            "A beautiful middle aged blonde female cyborg sks woman with army decoration in a white spaceship, wearing super detail futuristic chrome silver armor full body in a big wide background, clear facial features, cinematic, in the style of blade runner and alan moore, 35mm lens, accent lighting, global illumination, polaroid photo",
            "a photo of an sks woman as the commander of a starship",
            "a beautiful sks woman commanding the starship enterprise",
            "movie wallpaper of beautiful sks woman as the commander of starfleet",
            "incredible photograph of the beautiful sks woman as the commander of starfleet",
            "a photo of a young and smiling sks woman wearing a spacesuit",
        ]

    for i in range(400):
        seed = random.randint(0, 100000)
        seed_everything(seed)
        text_input = random.choice(text_inputs)
        generate_basic(text_input, outdir, seed = seed)