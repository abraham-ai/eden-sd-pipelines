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

    controlnet_img_dir = "/data/xander/Projects/cog/xander_eden_stuff/xander/assets/controlnet/garden"
    init_imgs = [os.path.join(controlnet_img_dir, f) for f in os.listdir(controlnet_img_dir)]

    low_t = random.choice([75, 100, 125])
    high_t = random.choice([150, 200, 250])

    #init_img = "/data/xander/Projects/cog/xander_eden_stuff/xander/assets/controlnet/architecture/eden_logo_transparent copy.png"
    init_img = random.choice(init_imgs)
    
    control_input_img = Image.open(init_img).convert("RGB")
    canny_img = cv2.Canny(np.array(control_input_img), low_t, high_t)[:, :, None]
    canny_img = np.concatenate([canny_img, canny_img, canny_img], axis=2)
    control_image = Image.fromarray(canny_img)

    # save the image:
    control_img_path = os.path.join(SD_PATH, "controlnet_input.jpg")
    control_image.save(control_img_path)

    args = StableDiffusionSettings(
        #ckpt = random.choice(checkpoint_options),
        mode = "generate",
        W = random.choice([1024+640]),
        H = random.choice([1024+640]),
        sampler = random.choice(["euler", "euler_ancestral"]),
        steps = 60,
        guidance_scale = random.choice([6,8,10]),
        upscale_f = random.choice([1.0, 1.0]),
        text_input = text_input,
        text_input_2 = text_input_2,
        seed = seed,
        n_samples = 1,
        lora_path = None,
        init_image_data = control_img_path,
        controlnet_conditioning_scale = random.choice([0.4, 0.5, 0.6, 0.7]),
        controlnet_path = "controlnet-canny-sdxl-1.0",
    )

    args.W, args.H = match_aspect_ratio(args.W * args.H, control_image)
    args.low_t = low_t
    args.high_t = high_t

    # resize control_input_img to match args.W, args.H
    control_input_img = control_input_img.resize((args.W, args.H), Image.LANCZOS)

    #name = f'{prefix}{args.text_input[:40]}_{os.path.basename(args.lora_path)}_{args.seed}_{int(time.time())}{suffix}'
    name = f'{prefix}{args.text_input[:40]}_{args.seed}_{int(time.time())}{suffix}'

    name = name.replace("/", "_")
    generator = make_images(args)

    for i, img in enumerate(generator):
        frame = f'{name}_{i}.jpg'
        control_frame = f'{name}_{i}_cond.jpg'
        os.makedirs(outdir, exist_ok = True)
        img.save(os.path.join(outdir, frame), quality=95)
        control_input_img.save(os.path.join(outdir, control_frame), quality=95)

    # save settings
    settings_filename = f'{outdir}/{name}.json'
    save_settings(args, settings_filename)


if __name__ == "__main__":

    
    outdir = "controlnet_garden"

    text_inputs += [
        "Decaying urban alleyway, graffiti-tagged walls converging into a glowing, neon-lit Chinatown archway.",
        "Lush rainforest, vines and ferns up close, leading to a distant waterfall veiled in mist.",
        "Medieval castle hallways, torchlit walls drawing into a grand throne gleaming under a chandelier.",
        "Sandy desert, dunes closeby framing a distant oasis with verdant palm trees.",
        "Cosmic space vortex, starry expanse narrowing into a vibrant wormhole center.",
        "Underwater cave, coral-laden entrance expanding into a deep blue abyss, shimmers of distant marine life.",
        "Ancient library, bookshelf walls drawing in to reveal a lone illuminated manuscript on a pedestal.",
        "Snow-clad mountains, frosted pines up close, guiding eyes to a serene, distant monastery.",
        "Cyberpunk cityscape, neon billboards nearby merging into a far-off levitating train zipping through skyscrapers.",
        "Fantasy forest, magical will-o'-the-wisps up close, guiding toward a hidden elfin palace bathed in moonlight.",
        "Mystical cavern, gem-studded walls leading to a distant radiant crystal cluster.",
        "Victorian street, gas-lit lamps revealing a remote, fog-veiled bridge over the Thames.",
        "Clockwork corridor, brass gears close up converging to an intricate pendulum heart in motion.",
        "Ancient Egyptian tomb, hieroglyphic walls narrowing into a distant golden sarcophagus.",
        "Rooftop perspective, cityscape edges leading to a central distant park, bathed in sunset.",
        "Apocalyptic wasteland, broken vehicles nearby merging into a far-off green haven.",
        "Golden wheat fields, nearby sheaves leading to a distant windmill against an azure sky.",
        "Cobbled streets of Rome, buildings on both sides pointing toward the Colosseum bathed in twilight.",
        "Spiraled seashell, textured edges narrowing into a distant, echoing, oceanic void.",
        "Steampunk airship dock, craft close up, pointing toward a floating city silhouette against a coppery horizon.",
            
    ]
        
    for i in range(150):
        seed = random.randint(0, 100000)
        #seed = i

        seed_everything(seed)

        text_input = random.choice(text_inputs)
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