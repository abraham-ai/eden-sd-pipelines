import argparse
import os
from tqdm import tqdm
from PIL import Image
from clip_interrogator import Interrogator, Config
import random

CAPTION_MODELS = {
    'blip-base': 'Salesforce/blip-image-captioning-base',   # 990MB
    'blip-large': 'Salesforce/blip-image-captioning-large', # 1.9GB
    'blip2-2.7b': 'Salesforce/blip2-opt-2.7b',              # 15.5GB
    'blip2-flan-t5-xl': 'Salesforce/blip2-flan-t5-xl',      # 15.77GB
    'git-large-coco': 'microsoft/git-large-coco',           # 1.58GB
}

clip_model_names = [
    'ViT-L-14/openai',
    'ViT-H-14/laion2b_s32b_b79k',
]

def caption_image(root_dir, f, ci, args):
    basename, ext = os.path.splitext(f)
    image = Image.open(os.path.join(root_dir, f))

    if args.fast:
        prompt = ci.interrogate_fast(image)
    else:
        prompt = ci.interrogate(image)

    if len(args.prepend_txt) > 0:
        prompt = args.prepend_txt + prompt

    with open(os.path.join(root_dir, f"{basename}.txt"), 'w') as f:
        f.write(prompt)


def caption_folder(args):
    error_folder = "errors"

    ci_config = Config(
        caption_model_name = 'blip-large',
        caption_max_length = 32,
        clip_model_name    = 'ViT-H-14/laion2b_s32b_b79k',
        chunk_size = 2048      # batch size for CLIP, use smaller for lower VRAM
    )
    ci = Interrogator(ci_config)

    for root_dir, _, files in os.walk(args.input_dir):
        random.shuffle(files)
        for f in tqdm(files):
            basename, ext = os.path.splitext(f)
            if ext in args.extensions:
                txt_path = os.path.join(root_dir, f"{basename}.txt")
                
                if os.path.exists(txt_path):
                    print("skipping!")
                    continue
                try:
                    caption_image(root_dir, f, ci, args)
                except Exception as e:
                    print(f"Error with {f}: {e}")
                    os.makedirs(error_folder, exist_ok=True)
                    os.rename(os.path.join(root_dir, f), os.path.join(error_folder, f))
                    try:
                        os.rename(os.path.join(root_dir, f"{basename}.pt"), os.path.join(error_folder, f"{basename}.pt"))
                    except:
                        pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('--extensions', type=str, nargs='+', default=['.jpg', '.png', '.jpeg'])
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--prepend_txt', type=str, default='')
    args = parser.parse_args()

    caption_folder(args)