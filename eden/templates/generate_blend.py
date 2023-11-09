import sys
sys.path.append('..')

import json
import time
import os
import random
from PIL import Image
import moviepy.editor as mpy

from settings import StableDiffusionSettings, _device
from generation import *
from prompts import text_inputs
from eden_utils import *
from pipe import pipe_manager

from PIL import Image
import math

def save_img_overview(img_list, save_path='overview.png', cell_size=1024):
    num_imgs = len(img_list)
    if num_imgs == 1:
        img_list[0].save(save_path)
        return

    # Resize images
    resized_imgs = [img.resize((int(img.width * min(cell_size / img.width, cell_size / img.height)), 
                                int(img.height * min(cell_size / img.width, cell_size / img.height)))) 
                    for img in img_list]

    # Find optimal grid dimensions
    min_area = float('inf')
    for h in range(1, int(math.sqrt(num_imgs)) + 1):
        w = math.ceil(num_imgs / h)
        area = w * h * cell_size * cell_size
        if area < min_area:
            min_area = area
            grid_width, grid_height = w, h

    # Create canvas
    canvas = Image.new('RGB', (grid_width * cell_size, grid_height * cell_size), 'black')

    # Place images on canvas
    for i, img in enumerate(resized_imgs):
        x_offset = (i % grid_width) * cell_size + (cell_size - img.width) // 2
        y_offset = (i // grid_width) * cell_size + (cell_size - img.height) // 2
        canvas.paste(img, (x_offset, y_offset))

    canvas.save(save_path)

def crop_to_square(img):
    w, h = img.size
    if w == h:
        return img

    if w > h:
        left = (w - h) // 2
        right = w - left
        top = 0
        bottom = h
    else:
        top = (h - w) // 2
        bottom = h - top
        left = 0
        right = w

    return img.crop((left, top, right, bottom))

def blend(
    init_image,
    outdir, 
    seed = int(time.time())):

    assert isinstance(init_image, list)
    assert len(init_image) > 1

    args = StableDiffusionSettings(
        mode = "generate",
        clip_interrogator_mode = "fast",
        text_input = "HD photo",
        W = random.choice([1024+512]),
        H = random.choice([1024+512]),
        sampler = random.choice(["euler", "euler_ancestral"]),
        steps = 40,
        guidance_scale = random.choice([6,8,10]),
        seed = seed,
        upscale_f = 1.0,
        init_image_strength = random.choice([0.0, 0.05, 0.1]),
    )

    pipe = pipe_manager.get_pipe(args)
    ip_adapter = pipe_manager.enable_ip_adapter()

    #args.text_input = clip_interrogate(args.ckpt, args.init_image, args.clip_interrogator_mode, CLIP_INTERROGATOR_MODEL_PATH)
    #del_clip_interrogator_models()

    c_sum, uc_sum, pc_sum, puc_sum = 0, 0, 0, 0
    source_imgs = []

    for i, image in enumerate(init_image):
        img = load_img(image, 'RGB')

        # get square centre crop:
        img = crop_to_square(img)
        
        source_imgs.append(img)
        c, uc, pc, puc = pipe_manager.ip_adapter.create_embeds(img, scale=1.0, negative_prompt=args.uc_text)

        c_sum += c
        uc_sum += uc
        pc_sum += pc
        puc_sum += puc

    # Average all the individual elements of embeds
    args.c   = c_sum / len(init_image)
    args.uc  = uc_sum / len(init_image)
    args.pc  = pc_sum / len(init_image)
    args.puc = puc_sum / len(init_image)

    name = f'remix_{args.seed}_{int(time.time())}_{args.ip_image_strength}_{args.text_input.replace(" ", "_")}'
    name = f'{args.init_image_strength:.2f}_{args.ip_image_strength:.2f}_{args.text_input.replace(" ", "_")}_{args.seed}'

    _, images_pil = generate(args)
    for i, img in enumerate(images_pil):
        frame = f'{name}_{i}.jpg'
        os.makedirs(outdir, exist_ok = True)
        img.save(os.path.join(outdir, frame), quality=95)

    # save the original images:
    #save_img_overview(source_imgs, save_path=os.path.join(outdir, f'{name}_source.jpg'))

    # Also save the original images separately:
    for i, img in enumerate(source_imgs):
        frame = f'{name}_source_{i}.jpg'
        img.save(os.path.join(outdir, frame), quality=95)

    # save settings
    settings_filename = f'{outdir}/{name}.json'
    save_settings(args, settings_filename)


if __name__ == "__main__":

    outdir = "results_blend"
    input_dir = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/init_imgs/01_great_inits"

    for i in range(200):
        seed = int(time.time())
        n = random.choice([2])

        # sample n random image paths from input_dir:
        all_img_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
        img_paths = random.sample(all_img_paths, n)

        blend(img_paths, outdir, seed=seed)

