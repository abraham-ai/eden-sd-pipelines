from dataclasses import dataclass, field
from typing import List
from PIL import Image
from pathlib import Path
import numpy as np
import os
import uuid
import torch

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
    #ckpt: str = "eden:eden-v1"
    ckpt: str = "sdxl-v1.0"

    # Lora / finetuning:
    lora_path: str = None
    lora_scale: float = 1.0
    lora_paths: str = None # optional list of lora paths for each img seed for real2real

    #precision: str = 'autocast'
    compile_unet: bool = False  # use toch.compile() to speed things up
    half_precision: bool = True
    activate_tileable_textures: bool = False
    gpu_info_verbose: bool = True  # if True, print GPU info to console

    # mode
    mode: str = "generate"
    
    # dimensions, quantity
    W: int = 768
    H: int = 768

    # sampler params
    sampler: str = "euler"
    steps: int = 45
    min_steps: int = 7  # low_n steps often give artifacts, so adopt a min-n-steps
    guidance_scale: float = 7.5
    
    # ddim_eta: float = 0.0
    C: int = 4
    f: int = 8   
    #dynamic_threshold: float = None
    #static_threshold: float = None
    upscale_f: float = 1.0   # when != 1.0, perform two stage generation (generate first, then upscale)

    # Watermark
    watermark_path: str = None
    
    # input_image
    init_image: Image = None
    init_image_data: str = None
    init_image_strength: float = 0.0
    #init_image_inpaint_mode: str = None # ["mean_fill", "edge_pad", "cv2_telea", "cv2_ns"]
    init_sample: str = None
    init_latent: str = None
    start_timestep: str = None # used to start at a specific timestep in the denoising loop (LatentBlending)
    
    # conditioning vectors:
    c: str = None   # force a specific prompt conditioning vector
    uc: str = None  # force a specific negative prompt conditioning vector
    pc: str = None  # force 
    puc: str = None
    pooled_prompt_embeds: str = None  # force a specific pooled prompt conditioning vector
    negative_pooled_prompt_embeds: str = None  # force a specific pooled negative prompt conditioning vector

    # mask
    # mask_image: Image = None
    # mask_image_data: str = None
    # mask_invert: bool = False
    # mask_brightness_adjust: float = 1.0
    # mask_contrast_adjust: float = 1.0

    # single generation
    text_input: str = "hello world" 
    text_input_2: str = None  # optional, second prompt (for txt-encoder2 in SDXL)
    uc_text: str = "nude, naked, poorly drawn face, ugly, tiling, out of frame, extra limbs, disfigured, deformed body, blurry, blurred, watermark, text, grainy, signature, cut off, draft"  # negative prompting
    seed: int = 0
    n_samples: int = 1

    # if mode is interpolate or animate (video)
    n_frames: int = 1
    loop: bool = False
    smooth: bool = True  
    n_film: int = 0
    fps: int = 9
    
    # interpolations
    interpolation_texts: List = field(default_factory=lambda: [])
    interpolation_seeds: List = field(default_factory=lambda: [])
    interpolation_init_images: List = field(default_factory=lambda: [])
    interpolation_init_images_use_img2txt: bool = False
    #interpolation_init_images_top_k: int = 1
    interpolation_init_images_power: float = 2.0
    interpolation_init_images_min_strength: float = 0.25
    interpolation_init_images_max_strength: float = 0.90
    easy_way: bool = False   # True causes errors in lerp
    save_distances_to_dir: str = None

    # # video feedback (not compatible with interpolations)
    # animation_mode: str = None  # ['2D', '3D', 'Video Input']
    # color_coherence: str = 'Match Frame 0 LAB' # [None, 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB']
    # init_video: str = None
    # extract_nth_frame: int = 1
    # turbo_steps: int = 3
    # previous_frame_strength: float = 0.65
    # previous_frame_noise: float = 0.02
    # contrast: float = 1.0
    # angle: float = 0
    # zoom: float = 0
    # translation: List = field(default_factory=lambda: [0, 0, 0])
    # rotation: List = field(default_factory=lambda: [0, 0, 0])

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
    latent_blending_skip_f: List = field(default_factory=lambda: [0.15, 0.7])  # What fraction of the denoising trajectory to skip ahead when using LatentBlending Trick (start and end values for each frame)
    
    # disk folder interaction:
    frames_dir: str = "."  # root folder of all the data for this generation
    save_phase_data: bool = False  # store metadata (conditioning c's and scales) for each frame, used for later upscaling
    save_distance_data: bool = False  # store distance plots (mostly used for debugging)
