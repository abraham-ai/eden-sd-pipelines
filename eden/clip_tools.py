import time
import os
import torch
import PIL.Image
from PIL import Image
import sys, os, time

from clip_interrogator import Interrogator, Config

ci = None

v1_model_names = [
    "runwayml/stable-diffusion-v1-5",
    "prompthero/openjourney-v2",
    "dreamlike-art/dreamlike-photoreal-2.0",
    "eden:eden-v1",
]

def load_ci(sd_model_name, force_reload=False, clip_model_path=None):
    global ci
    if ci is None or force_reload:

        print(f"Loading ViT-L-14/openai for clip_interpolator...")
        ci = Interrogator(Config(clip_model_path=clip_model_path, clip_model_name="ViT-L-14/openai")) # SD 1.x

        #print(f"Loading ViT-H-14/laion2b_s32b_b79k for clip_interpolator (SDv2.x models)...")
        #ci = Interrogator(Config(clip_model_path=clip_model_path, clip_model_name="ViT-H-14/laion2b_s32b_b79k"))  # SD 2.x
    
    return ci


def clip_interrogate(sd_ckpt_name, init_img, clip_interrogator_mode, clip_model_path=None):
    ci = load_ci(sd_ckpt_name, False, clip_model_path=clip_model_path)
    if clip_interrogator_mode == "fast":
        prompt = ci.interrogate_fast(init_img)
    else:
        prompt = ci.interrogate(init_img)
        
    prompt = prompt.replace("arafed", "")

    return prompt


def del_clip_interrogator_models():
    global ci
    try:
        del ci
    except:
        pass
    ci = None
    torch.cuda.empty_cache()


if __name__ == "__main__":
    init_img = Image.open("eden/assets/eden_logo.png")
    prompt = clip_interrogate("v1", init_img, "fast")
    print(prompt)