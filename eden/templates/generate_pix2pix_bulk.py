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


from diffusers import StableDiffusionInstructPix2PixPipeline
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.safety_checker = None

def generate_pix2pix(init_image, text_input, outdir, 
    steps_per_update = None, # None to disable intermediate frames
    seed = int(time.time()),
    debug = False):

    seed_everything(seed)
    init_image = Image.open(init_image).convert("RGB")
    args = StableDiffusionSettings(
        mode = "pix2pix",
        W = 640,
        H = 640,
    )

    args.W, args.H = match_aspect_ratio(args.W * args.H, init_image)
    args.init_image = init_image.resize((args.W, args.H), Image.LANCZOS)
    args.text_input = text_input
    args.image_guidance_scale = random.uniform(1.8, 2.25)


    pipe_output = pipe(
            prompt = text_input, 
            image = args.init_image,
            negative_prompt = args.uc_text,
            num_inference_steps = 70,
            image_guidance_scale = args.image_guidance_scale,
            guidance_scale = 7.5,
            num_images_per_prompt = 1,
            )


    # strip all / characters from text_input
    args.text_input = args.text_input.replace("/", "_")
    name = f'{args.text_input[:40]}_{seed}'

    for i, img in enumerate(pipe_output.images):
        fname = f"{name}_{i}_{args.image_guidance_scale:.3f}.jpg"
        os.makedirs(outdir, exist_ok = True)
        img.save(os.path.join(outdir, fname), quality=95)


if __name__ == "__main__":

    """

    export CUDA_VISIBLE_DEVICES=1
    cd /home/xander/Projects/cog/diffusers/eden/templates
    python3 generate_pix2pix_bulk.py

    """

    outdir = "results"
    input_img_dir = "/home/xander/Pictures/Mars2023/vince"

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
        "a long haired capybara",
        "big sir monster is a hybrid of shrek, big foot, elephant, and hippo",
        "Very very very very highly detailed epic central composition photo of Mr Bean as Shrek face, intricate, extremely detailed, digital painting, smooth, sharp focus, illustration, happy lighting, incredible art by Brooke Shaden, artstation, concept art, Octane render in Maya and Houdini",
        "shrek in a milk pool, 4 k, art station concept, highly detailed",
        "kevin james as shrek, movie still",
    ]
    
    text_inputs = [
        "make the man a caveman",
        "make the man look like a clown",
        "he looks like Neo from the matrix",
        "make the man look ridiculous",
        "make the man look like cookie monster",
        "make him look like a fat baby",
        "make him covered in hair and fur",
        "make him look like a horse, the head of a horse",
        "what would he look like if he was a donkey",
        "make him look like shrek, the green ogre",
        "make him look like a sumo wrestler",
        "make him look like a fat clown",
        "make him look like rusty statue",
        "turn his hair on fire",
        "what would it look like if his hair was on fire",
        "make him look really fat and hairy",
        "make him look like an angry caveman",
    ]

    for i in range(100):
        seed = int(time.time())
        seed_everything(seed)

        init_image = os.path.join(input_img_dir, random.choice(os.listdir(input_img_dir)))
        
        text_input = random.choice(text_inputs)
        generate_pix2pix(init_image, text_input, outdir, seed = seed)