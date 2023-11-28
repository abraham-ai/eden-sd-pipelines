from dataclasses import dataclass, field
from typing import List
from PIL import Image
from pathlib import Path
import numpy as np
import os
import uuid
import torch
import packaging.version
import transformers

print("-------- Default HF download path: ---------------")
print(transformers.file_utils.default_cache_path)
print("--------------------------------------------------")

if packaging.version.parse(torch.__version__) >= packaging.version.parse('1.12.0'):
    torch.backends.cuda.matmul.allow_tf32 = True

def pick_best_gpu_id():
    # pick the GPU with the most free memory:
    gpu_ids = [i for i in range(torch.cuda.device_count())]
    print(f"# of visible GPUs: {len(gpu_ids)}")
    gpu_mem = []
    for gpu_id in gpu_ids:
        free_memory, tot_mem = torch.cuda.mem_get_info(device=gpu_id)
        gpu_mem.append(free_memory)
        print("GPU %d: %d MB free" %(gpu_id, free_memory / 1024 / 1024))
    
    if len(gpu_ids) == 0:
        # no GPUs available, use CPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return None

    best_gpu_id = gpu_ids[np.argmax(gpu_mem)]
    # set this to be the active GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_id)
    print("Using GPU %d" %best_gpu_id)
    return best_gpu_id

gpu_id = pick_best_gpu_id()
global _device
if gpu_id is None:
    _device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
else:
    _device = torch.device("cuda:%d" %gpu_id if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(_device)

@dataclass
class StableDiffusionSettings:
    # unique identifier for this generation:
    uid: str = str(uuid.uuid4())
    
    # model settings
    ckpt: str = "juggernaut_XL2"
    upscale_ckpt: str = "juggernaut_XL2"   # "sdxl-refiner-v1.0"

    # controlnet
    control_image: Image = None
    control_image_path: str = None
    control_image_strength: float = 0.0
    control_guidance_start: float = 0.0
    control_guidance_end:   float = 1.0
    controlnet_path: str = None

    # Lora / finetuning:
    lora_path: str = None
    lora_scale: float = 0.7
    lora_paths: str = None # optional list of lora paths for each img seed for real2real

    #precision: str = 'autocast'
    compile_unet: bool = False  # use toch.compile() to speed things up
    half_precision: bool = True
    activate_tileable_textures: bool = False
    gpu_info_verbose: bool = True  # if True, print GPU info to console

    # mode
    mode: str = "generate"
    
    # dimensions, quantity
    W: int = 1024
    H: int = 1024
    n_target_pixels: int = None

    # sampler params
    sampler: str = "euler"
    steps: int = 40
    min_steps: int = 7  # low_n steps often give artifacts, so adopt a min-n-steps
    guidance_scale: float = 7.5
    use_lcm: bool = False
    
    C: int = 4
    f: int = 8   
    upscale_f: float = 1.0   # when != 1.0, perform two stage generation (generate first, then upscale)

    # Watermark
    watermark_path: str = None
    
    # input_image
    init_image: str = None
    init_image_path: str = None
    init_image_strength: float = 0.0
    adopt_aspect_from_init_img: bool = True
    
    init_sample: str = None
    init_latent: str = None
    start_timestep: str = None # used to start at a specific timestep in the denoising loop (LatentBlending)
    
    # conditioning vectors:
    ip_image: str = None
    ip_image_path: str = None
    ip_image_strength: float  = 0.65     # 1.0 will only use the image prompt, 0.0 will only use the text prompt

    c: str = None   # force a specific prompt conditioning vector
    uc: str = None  # force a specific negative prompt conditioning vector
    pc: str = None  # force 
    puc: str = None
    pooled_prompt_embeds: str = None  # force a specific pooled prompt conditioning vector
    negative_pooled_prompt_embeds: str = None  # force a specific pooled negative prompt conditioning vector

    # single generation
    name: str = "" # prompt-name of the creation in the UI
    text_input: str = "hello world" 
    text_input_2: str = None  # optional, second prompt (for txt-encoder2 in SDXL)
    #uc_text: str = "nude, naked, poorly drawn face, ugly, tiling, out of frame, extra limbs, disfigured, deformed body, blurry, blurred, watermark, text, grainy, signature, cut off, draft"  # negative prompting
    uc_text: str = "nude, naked, text, watermark, low-quality, signature, padding, margins, white borders, padded border, moiré pattern, downsampling, aliasing, distorted, blurry, blur, jpeg artifacts, compression artifacts, poorly drawn, low-resolution, bad, grainy, error, bad-contrast"
    seed: int = 0
    noise_sigma: float = 0.0 # adds random noise to the conditioning vector (values of around 0.1 - 0.5 are OK)
    n_samples: int = 1

    # if mode is interpolate or animate (video)
    n_frames: int = 1
    loop: bool = False
    smooth: bool = True  
    n_film: int = 0
    fps: int = 12
    
    # interpolations
    interpolation_texts: List = field(default_factory=lambda: [])
    interpolation_seeds: List = field(default_factory=lambda: [])
    interpolation_init_images: List = field(default_factory=lambda: [])
    interpolation_init_images_power: float = 2.5
    interpolation_init_images_min_strength: float = 0.05 # SDXL is very sensitive to init_imgs
    interpolation_init_images_max_strength: float = 0.80
    save_distances_to_dir: str = None

    # personalized aesthetic gradients:
    aesthetic_target: List = field(default_factory=lambda: None)   # either a path to a .pt file, or a list of PIL.Image objects
    aesthetic_steps: int          = 0
    aesthetic_lr: float           = 0.0001
    ag_L2_normalization_constant: float = 0.05

    # img2txt:
    clip_interrogator_mode: str = "fast" # ["full", "fast"]

    # audio modulation:
    planner: str = None

    # Latent Tracking:
    interpolator: str = None
    n_anchor_imgs: int = 3  # number of anchor images to render before starting latent blending
    latent_blending_skip_f: List = field(default_factory=lambda: [0.05, 0.65])  # What fraction of the denoising trajectory to skip ahead when using LatentBlending Trick (start and end values for each frame)
    lpips_max_d: float = 0.70  # once interframe lpips distance dips below this value, blending_skip_f will start increasing from latent_blending_skip_f[0]
    lpips_min_d: float = 0.05  # once interframe lpips distance dips below this value, blending_skip_f will saturate at latent_blending_skip_f[1]

    never_overwrite_existing_latents: bool = True # if True, will never overwrite real, existing latents in the tracker

    # disk folder interaction:
    frames_dir: str = "."  # root folder of all the data for this generation
    save_phase_data: bool = False  # store metadata (conditioning c's and scales) for each frame, used for later upscaling
    save_distance_data: bool = False  # store distance plots (mostly used for debugging)
