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

def generate_basic(
    text_input, 
    outdir, 
    steps_per_update = None, # None to disable intermediate frames
    text_input_2 = None,
    seed = int(time.time()),
    debug = False,
    init_image_data = None,
    prefix = "",
    suffix = ""):


    init_imgs = [None]

    #lora_tar_path = "https://pbxt.replicate.delivery/ZCxKsJNhsH7uMxAfM4PeD4mGMMEo9nQlbP3YFwgCnHSrke5iA/trained_model.tar"
    #lora_name     = "banny_all"
    #lora_path = download_an_untar(lora_tar_path, "replicate_tar_loras", lora_name)

    args = StableDiffusionSettings(
        #ckpt = random.choice(checkpoint_options),
        mode = "generate",
        W = random.choice([1024]),
        H = random.choice([1024]),
        sampler = random.choice(["euler"]),
        steps = 60,
        guidance_scale = random.choice([6,8,10]),
        upscale_f = random.choice([1.0, 1.0]),
        text_input = text_input,
        text_input_2 = text_input_2,
        seed = seed,
        n_samples = 1,
        lora_path = None,
        #lora_path = lora_path,
        #lora_path = "/data/xander/Projects/cog/xander_eden_stuff/loras/diffusers/banny1/pytorch_lora_weights.bin",
        #lora_path = "/data/xander/Projects/cog/diffusers/lora/lora_compare_diffusers/gene_single/checkpoint-400",
        #lora_scale = random.choice([0.8]),
        #init_image_data = "/data/xander/Projects/cog/eden-sd-pipelines/eden/templates/garden_inputs/a closeup portrait of a woman wrapped in_203_1691967507_0.jpg",
        #init_image_strength = random.choice([0.1, 0.15, 0.2, 0.25, 0.3, 0.35]),
    )

    #name = f'{prefix}{args.text_input[:40]}_{os.path.basename(args.lora_path)}_{args.seed}_{int(time.time())}{suffix}'
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
        outdir = "controlnet"

        json_dir = "/data/xander/Projects/cog/xander_eden_stuff/xander/assets/hetty/templates2"
        # load all the json files from the json_dir:
        json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if ".json" in f]

        # for each json_file, load the "text_input" field:
        text_inputs = []
        for json_file in json_files:
            with open(json_file, "r") as f:
                j = json.load(f)
                text_inputs.append(j["text_input"])

        extra_prompts = [
            "a photo of Cate Blanchett as the commander of the starfleet enterprise",
            "a photo of Cate Blanchett as the commander of starfleet",
            "a photo of Cate Blanchett as the commander of the starfleet, posing proud in the hallway of her spaceship",
            "a photo of Cate Blanchett standing in the control room of the galactic starship, posing proud in front her crew who's blurry in the background, smiling, leader figure",
            "A photo of Cate Blanchett in the captain's chair of a futuristic starship, her crew diligently working in the background",
            "An image of Cate Blanchett dressed as the admiral of a galactic fleet, standing tall and authoritative on the bridge of her flagshig",
            "A picture of Cate Blanchett as the supreme commander of a space armada, surveying her fleet from the observation deck, determination in her eyes",
            "A still of Cate Blanchett portraying the role of a space captain, confidently guiding her starship through the unexplored territories of the galaxy",
            "A snapshot of Cate Blanchett in a commander's uniform, standing at the helm of an interstellar cruiser, her face reflecting the weight of leadership",
            "An illustration of Cate Blanchett as the general of an advanced space force, briefing her team in the war room, a holographic star map glowing before them",
            "A photograph of Cate Blanchett, poised and dignified as the head of a space exploration mission, in the control room with her dedicated crew blurred in the background",
            "A cinematic shot of Cate Blanchett as a starship captain, saluting her fleet from the command deck, with the endless cosmos stretching out behind her",
            "A rendering of Cate Blanchett in a futuristic uniform, strategizing with her officers in the operations room of a cutting-edge galactic vessel",
            "A freeze-frame of Cate Blanchett, leader of an elite space corps, standing in front of her soldiers, with the glimmering stars and planets visible through the viewport behind her",
        ]
        text_inputs.extend(extra_prompts)

        # replace <person1> with "Cate Blanchett" in each of the text_inputs:
        text_inputs = [t.replace("<person1>", "Cate Blanchett") for t in text_inputs]
        text_inputs = [t.replace("dystopian", "utopian") for t in text_inputs]

        #text_inputs = ['a photo of <s0><s1> as the commander of the starfleet enterprise']

    else:
        outdir = "plantoid"
        
        text_inputs = [
            "a cartoon of TOK as a superhero",
            "a cartoon of TOK on a surfboard riding a wave",
            "a photo of TOK climbing mount everest",
            "a masterpiece artwork of TOK",
            "a photo of a massive TOK statue",
            "a photo of TOK exploring the Amazon rainforest",
            "a photo of TOK at the top of the Eiffel Tower",
            "a photo of a TOK-themed amusement park",
            "a photo of TOK meditating, reaching enlightenment",
            ]
    
    for i in range(20):
        n_modifiers = random.randint(0, 3)
        #seed = random.randint(0, 100000)
        seed = i

        seed_everything(seed)
        #text_input = random.choice(text_inputs)
        text_input = text_inputs[i%len(text_inputs)]

        #text_input = text_input.replace("TOK", "<s0><s1>")
        text_input = text_input.replace("TOK", "plantoid")
        #text_input = text_input.replace("TOK", "Banny")

        if n_modifiers > 0:
            text_input = text_input + ", " + ", ".join(random.sample(modifiers, n_modifiers))

        #text_input = text_inputs[i%len(text_inputs)]
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