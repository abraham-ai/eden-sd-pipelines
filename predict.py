import os
import sys
import tempfile
import random
import requests
import hashlib
from typing import Iterator, Optional
import moviepy.editor as mpy
import numpy as np
from dotenv import load_dotenv

load_dotenv()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_HOME"] = "/src/.torch"
os.environ["TRANSFORMERS_CACHE"] = "/src/.huggingface/"
os.environ["DIFFUSERS_CACHE"] = "/src/.huggingface/"
os.environ["HF_HOME"] = "/src/.huggingface/"
os.environ["LPIPS_HOME"] = "/src/models/lpips/"

sys.path.extend([
    "./eden",
    "/clip-interrogator",
    "/frame-interpolation",
    "./lora",
    "./lora/lora_diffusion"
])

from settings import StableDiffusionSettings
import eden_utils
import film
from lora import train_lora

from cog import BasePredictor, BaseModel, File, Input, Path

checkpoint_options = [
    "runwayml/stable-diffusion-v1-5",
    "prompthero/openjourney-v2",
    "dreamlike-art/dreamlike-photoreal-2.0"
]


class CogOutput(BaseModel):
    file: Path
    name: Optional[str] = None
    thumbnail: Optional[Path] = None
    attributes: Optional[dict] = None
    progress: Optional[float] = None
    isFinal: bool = False


def download(url, folder, ext):
    filename = url.split('/')[-1]+ext
    filepath = folder / filename
    if filepath.exists():
        return filepath
    raw_file = requests.get(url, stream=True).raw
    with open(filepath, 'wb') as f:
        f.write(raw_file.read())
    return filepath


class Predictor(BasePredictor):

    def setup(self):
        print("cog:setup")
        import generation
        import interpolator
        generation.CLIP_INTERROGATOR_MODEL_PATH = '/src/cache'
        interpolator.LPIPS_DIR = "/src/models/lpips/weights/v0.1/alex.pth"
        film.FILM_MODEL_PATH = "/src/models/film/film_net/Style/saved_model"

    def predict(
        self,
        
        # Universal args
        mode: str = Input(
            description="Mode", default="generate",
            choices=["generate", "remix", "interpolate", "real2real", "interrogate", "lora"]
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
            ge=64, le=2048, default=512
        ),
        height: int = Input(
            description="Height", 
            ge=64, le=2048, default=512
        ),
        checkpoint: str = Input(
            description="Which Stable Diffusion checkpoint to use",
            choices=checkpoint_options,
            default="dreamlike-art/dreamlike-photoreal-2.0"
        ),
        lora: str = Input(
            description="(optional) URL of Lora finetuning",
            default=None
        ),
        lora_scale: float = Input(
            description="Lora scale (how much of the Lora finetuning to apply)",
            ge=0.0, le=1.0, default=0.8
        ),
        sampler: str = Input(
            description="Which sampler to use", 
            default="euler", 
            # choices=["ddim", "plms", "klms", "dpm2", "dpm2_ancestral", "heun", "euler", "euler_ancestral"]
            choices=["euler"]
        ),
        steps: int = Input(
            description="Diffusion steps", 
            ge=0, le=200, default=60
        ),
        guidance_scale: float = Input(
            description="Strength of text conditioning guidance", 
            ge=0, le=32, default=10.0
        ),
        upscale_f: int = Input(
            description="Upscaling resolution",
            ge=1, le=4, default=1
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

        # Generate mode
        text_input: str = Input(
            description="Text input (mode==generate)",
        ),
        uc_text: str = Input(
            description="Negative text input (mode==all)",
            default="poorly drawn face, ugly, tiling, out of frame, extra limbs, disfigured, deformed body, blurry, blurred, watermark, text, grainy, signature, cut off, draft"
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
            ge=0, le=100, default=50
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
            ge=0.0, le=8.0, default=2.5
        ),
        interpolation_init_images_min_strength: float = Input(
            description="Minimum init image strength for interpolation_init_images prompts (mode==interpolate)",
            ge=0, le=1, default=0.2
        ),
        scale_modulation: float = Input(
            description="Scale modulation amplitude for interpolation (mode==interpolate)",
            ge=0.0, le=10, default=0.0
        ),
        loop: bool = Input(
            description="Loops (mode==interpolate)",
            default=True
        ),
        smooth: bool = Input(
            description="Smooth (mode==interpolate)",
            default=False
        ),
        n_film: int = Input(
            description="Number of times to smooth final frames with FILM (default is 0) (mode==interpolate)",
            default=0, ge=0, le=2
        ),
        fps: int = Input(
            description="Frames per second (mode==interpolate)",
            default=12, ge=1, le=60
        ),
        
        # Lora
        lora_training_urls: str = Input(
            description="Training images for new LORA concept (mode==lora)", 
            default=None
        ),

    ) -> Iterator[CogOutput]:

        print("cog:predict:")
        
        import generation
        
        interpolation_texts = interpolation_texts.split('|') if interpolation_texts else None
        interpolation_seeds = [float(i) for i in interpolation_seeds.split('|')] if interpolation_seeds else None
        interpolation_init_images = interpolation_init_images.split('|') if interpolation_init_images else None

        lora_path = None
        if lora:
            lora_path = download(lora, 'loras', '.safetensor')

        args = StableDiffusionSettings(
            ckpt = checkpoint,
            lora_path = None,

            mode = mode,

            W = width - (width % 64),
            H = height - (height % 64),
            sampler = sampler,
            steps = steps,
            guidance_scale = guidance_scale,
            upscale_f = float(upscale_f),

            init_image_data = init_image_data,
            init_image_strength = init_image_strength,

            text_input = text_input,
            uc_text = uc_text,
            seed = seed,
            n_samples = n_samples,

            interpolation_texts = interpolation_texts,
            interpolation_seeds = interpolation_seeds,
            interpolation_init_images = interpolation_init_images,
            interpolation_init_images_power = interpolation_init_images_power,
            interpolation_init_images_min_strength = interpolation_init_images_min_strength,

            n_frames = n_frames,
            scale_modulation = scale_modulation,
            loop = loop,
            smooth = smooth,
            n_film = n_film,
            fps = fps,

            lora_path = lora_path,
            lora_scale = lora_scale,

            aesthetic_target = None, # None means we'll use the init_images as target
            aesthetic_steps = 10,
            aesthetic_lr = 0.0001,
            ag_L2_normalization_constant = 0.25, # for real2real, only 
        )

        print(args)

        out_dir = Path(tempfile.mkdtemp())

        if mode == "lora":
            data_dir = Path(tempfile.mkdtemp())
            data_dir.mkdir(exist_ok=True)
            lora_training_urls = lora_training_urls.split('|')
            for lora_url in lora_training_urls:
                lora_file = download(lora_url, data_dir, '.jpg')

            train_lora(
                instance_data_dir = str(data_dir),
                pretrained_model_name_or_path = checkpoint,
                output_dir = str(out_dir),
                out_name = "final_lora",
                train_text_encoder = True,
                perform_inversion = True,
                resolution = 512,
                train_batch_size = 4,
                gradient_accumulation_steps = 1,
                scale_lr = True,
                learning_rate_ti = 2.5e-4,
                continue_inversion = True,
                continue_inversion_lr = 2.5e-5,
                learning_rate_unet = 1.5e-5,
                learning_rate_text = 2.5e-5,
                color_jitter = True,
                lr_scheduler = "linear",
                lr_warmup_steps = 0,
                placeholder_tokens = "<person1>",
                proxy_token = "person",
                use_template = "person",
                use_mask_captioned_data = False,
                save_steps = 500,
                max_train_steps_ti = 300,
                max_train_steps_tuning = 500,
                clip_ti_decay = True,
                weight_decay_ti = 0.0005,
                weight_decay_lora = 0.001,
                lora_rank_unet = 2,
                lora_rank_text_encoder  =8,
                cached_latents = False,
                use_extended_lora = False,
                enable_xformers_memory_efficient_attention = True,
                use_face_segmentation_condition = True,
                device = "cuda:0"
            )

            lora_location = out_dir / 'final_lora.safetensors'
            print(os.system(f'ls {str(lora_location)}'))

            yield CogOutput(file=lora_location, name="final_lora", thumbnail=None, attributes=None, isFinal=True, progress=1.0)

        elif mode == "interrogate":
            interrogation = generation.interrogate(args)
            out_path = out_dir / f"interrogation.txt"
            with open(out_path, 'w') as f:
                f.write(interrogation)
            attributes = {'interrogation': interrogation}
            yield CogOutput(file=out_path, name=interrogation, thumbnail=None, attributes=attributes, isFinal=True, progress=1.0)

        elif mode == "generate" or mode == "remix":
            frames = generation.make_images(args)
            frame = frames[0]  # just one frame for now

            attributes = None
            if mode == "remix":
                attributes = {"interrogation": args.text_input}

            name = args.text_input
            out_path = out_dir / f"frame.jpg"
            frame.save(out_path, format='JPEG', subsampling=0, quality=95)
            
            yield CogOutput(file=out_path, name=name, thumbnail=out_path, attributes=attributes, isFinal=True, progress=1.0)
            
        else:
            
            if mode == "interpolate":
                generator = generation.make_interpolation(args)

            elif mode == "real2real":
                args.interpolation_init_images_use_img2txt = True
                generator = generation.make_interpolation(args)

            attributes = None
            thumbnail = None

            # generate frames
            for f, (frame, t_raw) in enumerate(generator):
                out_path = out_dir / ("frame_%0.16f.jpg" % t_raw)
                frame.save(out_path, format='JPEG', subsampling=0, quality=95)
                progress = f / args.n_frames
                if not thumbnail:
                    thumbnail = out_path
                if stream and f % stream_every == 0:
                   yield CogOutput(file=out_path, thumbnail=None, attributes=attributes, progress=progress)

            # run FILM
            if args.n_film > 0:
                film.interpolate_FILM(str(out_dir), n_film)
                out_dir = out_dir / "interpolated_frames"

            # save video
            loop = (args.loop and len(args.interpolation_seeds) == 2)
            out_path = out_dir / "out.mp4"
            eden_utils.write_video(out_dir, str(out_path), loop=loop, fps=args.fps)

            if mode == "real2real":
                attributes = {"interrogation": args.interpolation_texts}

            name = " => ".join(args.interpolation_texts)

            yield CogOutput(file=out_path, name=name, thumbnail=thumbnail, attributes=attributes, isFinal=True, progress=1.0)
