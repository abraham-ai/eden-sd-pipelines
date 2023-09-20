# don't push DEBUG_MODE = True to Replicate!
DEBUG_MODE = False
#DEBUG_MODE = True

from pathlib import Path
import os
import time
import random
import sys
import json
import tempfile
import requests
import subprocess
import signal
from typing import Iterator, Optional
from dotenv import load_dotenv
from copy import deepcopy

load_dotenv()

os.environ["TORCH_HOME"] = "/src/.torch"
os.environ["TRANSFORMERS_CACHE"] = "/src/.huggingface/"
os.environ["DIFFUSERS_CACHE"] = "/src/.huggingface/"
os.environ["HF_HOME"] = "/src/.huggingface/"
os.environ["LPIPS_HOME"] = "/src/models/lpips/"

sys.path.extend([
    "./eden",
    "./lora",
    "./lora/lora_diffusion",
    "/clip-interrogator",
])

from io_utils import *
from settings import StableDiffusionSettings
import eden_utils
from cog import BasePredictor, BaseModel, File, Input, Path

checkpoint_options = [
    "sdxl-v1.0",
]
checkpoint_default = "sdxl-v1.0"

class CogOutput(BaseModel):
    files: Optional[list[Path]] = []
    name: Optional[str] = None
    thumbnails: Optional[list[Path]] = []
    attributes: Optional[dict] = None
    progress: Optional[float] = None
    isFinal: bool = False

class Predictor(BasePredictor):

    GENERATOR_OUTPUT_TYPE = Path if DEBUG_MODE else CogOutput

    def setup(self):
        print("cog:setup")
        import generation
        import interpolator
        generation.CLIP_INTERROGATOR_MODEL_PATH = '/src/cache'
        interpolator.LPIPS_DIR = "/src/models/lpips/weights/v0.1/alex.pth"

    def predict(
        self,
        
        # Universal args
        mode: str = Input(
            description="Mode", default="generate",
            choices=["generate", "remix", "upscale", "blend", "controlnet", "interpolate", "real2real", "interrogate"]
        ),
        stream: bool = Input(
            description="yield individual results if True", default=False
        ),
        stream_every: int = Input(
            description="for mode generate, how many steps per update to stream (steam must be set to True)", 
            default=1, ge=1, le=25
        ),
        width: int = Input(
            description="Width", 
            ge=512, le=2048, default=1024
        ),
        height: int = Input(
            description="Height", 
            ge=512, le=2048, default=1024
        ),
        checkpoint: str = Input(
            description="Which Stable Diffusion checkpoint to use",
            choices=checkpoint_options,
            default=checkpoint_default
        ),
        lora: str = Input(
            description="(optional) URL of Lora finetuning",
            default=None
        ),
        lora_scale: float = Input(
            description="Lora scale (how much of the Lora finetuning to apply)",
            ge=0.0, le=1.2, default=0.8
        ),
        sampler: str = Input(
            description="Which sampler to use", 
            default="euler", 
            choices=["ddim", "ddpm", "klms", "euler", "euler_ancestral", "dpm", "kdpm2", "kdpm2_ancestral", "pndm"]
        ),
        steps: int = Input(
            description="Diffusion steps", 
            ge=10, le=100, default=35
        ),
        guidance_scale: float = Input(
            description="Strength of text conditioning guidance", 
            ge=1, le=20, default=7.5
        ),
        upscale_f: float = Input(
            description="Upscaling resolution",
            ge=1, le=2, default=1
        ),

        # Init image and mask
        init_image_data: str = Input(
            description="Load initial image from file, url, or base64 string", 
            default=None
        ),
        init_image_strength: float = Input(
            description="Strength of initial image", 
            ge=0.0, le=1.0, default=0.0
        ),
        adopt_aspect_from_init_img: bool = Input(
            description="Adopt aspect ratio from init image",
            default=True
        ),
        controlnet_type: str = Input(
            description="Controlnet type",
            default="off",
            choices=["off", "canny-edge", "depth"]
        ),
        # Generate mode
        text_input: str = Input(
            description="Text input (mode==generate)", default=None
        ),
        uc_text: str = Input(
            description="Negative text input (mode==all)",
            default="watermark, text, nude, naked, nsfw, poorly drawn face, ugly, tiling, out of frame, blurry, blurred, grainy, signature, cut off, draft"
        ),
        seed: int = Input(
            description="random seed (mode==generate)", 
            ge=0, le=1e10, default=13
        ),
        n_samples: int = Input(
            description="batch size (mode==generate)",
            ge=1, le=4, default=1
        ),

        # Interpolate mode
        n_frames: int = Input(
            description="Total number of frames (mode==interpolate)",
            ge=3, le=1000, default=40
        ),

        # Interpolate mode
        interpolation_texts: str = Input(
            description="Interpolation texts (mode==interpolate)",
            default=None
        ),
        interpolation_seeds: str = Input(
            description="Seeds for interpolated texts (mode==interpolate)",
            default=None
        ),
        interpolation_init_images: str = Input(
            description="Interpolation init images, file paths or urls (mode==interpolate)",
            default=None
        ),
        interpolation_init_images_power: float = Input(
            description="Power for interpolation_init_images prompts (mode==interpolate)",
            ge=0.5, le=5.0, default=2.5
        ),
        interpolation_init_images_min_strength: float = Input(
            description="Minimum init image strength for interpolation_init_images prompts (mode==interpolate)",
            ge=0, le=1.0, default=0.25
        ),
        interpolation_init_images_max_strength: float = Input(
            description="Maximum init image strength for interpolation_init_images prompts (mode==interpolate)",
            ge=0.0, le=1.0, default=0.95
        ),
        loop: bool = Input(
            description="Loops (mode==interpolate)",
            default=True
        ),
        smooth: bool = Input(
            description="Smooth (mode==interpolate)",
            default=True
        ),
        latent_blending_skip_f: str = Input(
            description="What fraction of the denoising trajectory to skip at the start and end of each interpolation phase, two floats, separated by a pipe (|)",
            default="0.15|0.65"
        ),
        n_film: int = Input(
            description="Number of times to smooth final frames with FILM (default is 0) (mode==interpolate)",
            default=1, ge=0, le=3
        ),
        fps: int = Input(
            description="Frames per second (mode==interpolate)",
            default=12, ge=1, le=30
        ),

    ) -> Iterator[GENERATOR_OUTPUT_TYPE]:
    
        print("cog:predict:")
        import generation

        t_start = time.time()

        interpolation_texts = interpolation_texts.split('|') if interpolation_texts else None
        interpolation_seeds = [float(i) for i in interpolation_seeds.split('|')] if interpolation_seeds else None
        interpolation_init_images = interpolation_init_images.split('|') if interpolation_init_images else None
        
        lora_path = None
        if lora:
            lora_folder = Path('loras')
            lora_zip_path = download(lora, lora_folder)
            lora_path = os.path.join(lora_folder, os.path.splitext(os.path.basename(lora_zip_path))[0])
            extract_to_folder(lora_zip_path, lora_path)

        controlnet_options = {
            "off": None,
            "canny-edge": "controlnet-canny-sdxl-1.0-small",
            "depth":      "controlnet-depth-sdxl-1.0-small"
        }
        
        args = StableDiffusionSettings(
            ckpt = checkpoint,
            lora_path = lora_path,
            lora_scale = lora_scale,

            mode = mode,

            W = width - (width % 64),
            H = height - (height % 64),
            sampler = sampler,
            steps = steps,
            guidance_scale = guidance_scale,
            upscale_f = float(upscale_f),

            init_image_data = init_image_data,
            init_image_strength = init_image_strength,
            adopt_aspect_from_init_img = adopt_aspect_from_init_img,
            controlnet_path = controlnet_options[controlnet_type],

            text_input = text_input,
            uc_text = uc_text,
            seed = seed,
            n_samples = n_samples,

            interpolation_texts = interpolation_texts,
            interpolation_seeds = interpolation_seeds,
            interpolation_init_images = interpolation_init_images,
            interpolation_init_images_power = interpolation_init_images_power,
            interpolation_init_images_min_strength = interpolation_init_images_min_strength,
            interpolation_init_images_max_strength = interpolation_init_images_max_strength,

            latent_blending_skip_f = [float(i) for i in latent_blending_skip_f.split('|')],

            n_frames = n_frames,
            loop = loop,
            smooth = smooth,
            n_film = n_film,
            fps = fps,

            aesthetic_target = None, # None means we'll use the init_images as target
            aesthetic_steps = 10,
            aesthetic_lr = 0.0001,
            ag_L2_normalization_constant = 0.25, # for real2real, only 
        )
        
        print("Arguments:")
        print(args)
        print("--------------------------------------")

        out_dir = Path(tempfile.mkdtemp())

        if mode == "interrogate":
            interrogation = generation.interrogate(args)
            out_path = out_dir / f"interrogation.txt"
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(interrogation)
            attributes = {'interrogation': interrogation}
            if DEBUG_MODE:
                yield out_path
            else:
                yield CogOutput(files=[out_path], name=interrogation, thumbnails=[out_path], attributes=attributes, isFinal=True, progress=1.0)
        
        elif mode == "generate" or mode == "remix" or mode == "controlnet" or mode == "upscale":

            if (mode == "upscale" or mode == "remix" or mode == "controlnet") and (args.init_image_data is None):
                raise ValueError(f"an init_image must be provided for mode = {mode}")
            
            if args.init_image_data is None:
                args.init_image_strength = 0.0

            attributes = None
            print("Input prompt:")
            print(args.text_input)
            out_paths = []

            # slight overhead here to do iterative batching (avoid OOM):
            n_samples = args.n_samples
            args.n_samples = 1

            for batch_i in range(n_samples):
                batch_i_args = deepcopy(args)
                batch_i_args.seed += batch_i
                frames = generation.make_images(batch_i_args)
                for f, frame in enumerate(frames):
                    out_path = out_dir / f"frame_{f:04d}_{batch_i}.jpg"
                    frame.save(out_path, format='JPEG', subsampling=0, quality=95)
                    out_paths.append(out_path)
            
            if (mode == "remix" or mode == "upscale") and (args.text_input is None):
                attributes = {"interrogation": batch_i_args.text_input}
            
            if DEBUG_MODE:
                yield out_paths[0]
            else:
                yield CogOutput(files=out_paths, name=batch_i_args.name, thumbnails=out_paths, attributes=attributes, isFinal=True, progress=1.0)

        else: # mode == "interpolate" or mode == "real2real" or mode == "blend"

            if args.interpolation_seeds is None:
                # create random seeds with the same length as the number of texts / images:
                if mode == "interpolate":
                    n_keyframes = len(args.interpolation_texts)
                else:
                    n_keyframes = len(args.interpolation_init_images)
                args.interpolation_seeds = [random.randint(0, 9999) for _ in range(n_keyframes)]

            if mode == "blend":
                assert len(args.interpolation_init_images) == 2, "Must have exactly two init_images to blend!"
                args.n_frames = 5
                args.n_film = 0
                args.smooth = True
                args.loop = False
                args.interpolation_init_images_max_strength = 1.0
                #force_timepoints = [0.0, 1.0, 0.25] # TODO enable weighted blending
                force_timepoints = None

            else:
                force_timepoints = None

            # Make sure there's at least two init_images or prompts to interpolate:
            if (mode == "interpolate" and len(args.interpolation_texts) < 2) or (mode == "real2real" and len(args.interpolation_init_images) < 2):
                raise ValueError("Must have at least two init_images or prompts to interpolate!")

            loop = (args.loop and len(args.interpolation_seeds) == 2)

            if loop:
                args.n_frames = args.n_frames // 2
            
            generator = generation.make_interpolation(args, force_timepoints=force_timepoints)
            attributes = None
            thumbnail = None

            # generate frames
            for f, (frame, t_raw) in enumerate(generator):
                out_path = out_dir / ("frame_%0.16f.jpg" % t_raw)
                frame.save(out_path, format='JPEG', subsampling=0, quality=95)

                if not thumbnail:
                    thumbnail = out_path

                if (mode == "blend") and (t_raw == 0.5):
                    thumbnail = out_path
                    print("predict.py: blend mode, saving frame 0.5")
                    break

                progress = f / args.n_frames
                cog_output = CogOutput(attributes=attributes, progress=progress)
                if stream and f % stream_every == 0:
                    cog_output.files = [out_path]
                yield cog_output

            # run FILM
            if args.n_film > 0:
                print('predict.py: running FILM...')
                FILM_MODEL_PATH = "/src/models/film/film_net/Style/saved_model"
                abs_out_dir_path = os.path.abspath(str(out_dir))
                command = ["python", "/src/eden/film.py", "--frames_dir", abs_out_dir_path, "--times_to_interpolate", str(args.n_film), '--update_film_model_path', FILM_MODEL_PATH]
                
                run_and_kill_cmd(command)
                print("predict.py: FILM done.")
                out_dir = Path(os.path.join(abs_out_dir_path, "interpolated_frames"))

            if mode != "blend":
                # save video
                out_path = out_dir / "out.mp4"
                eden_utils.write_video(out_dir, str(out_path), loop=loop, fps=args.fps)

                if mode == "real2real":
                    attributes = {"interrogation": args.interpolation_texts}

            if DEBUG_MODE:
                shutil.copyfile(out_path, "/src/out.jpg")
                yield out_path
            else:
                yield CogOutput(files=[out_path], name=args.name, thumbnails=[thumbnail], attributes=attributes, isFinal=True, progress=1.0)

        t_end = time.time()
        print(f"predict.py: done in {t_end - t_start:.2f} seconds")
