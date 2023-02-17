import os
import sys
import tempfile
import random
import hashlib
from typing import Iterator, Optional
import moviepy.editor as mpy
import numpy as np
from dotenv import load_dotenv

load_dotenv()

os.environ["TORCH_HOME"] = "/src/.torch"
os.environ["TRANSFORMERS_CACHE"] = "/src/.huggingface/"
os.environ["DIFFUSERS_CACHE"] = "/src/.huggingface/"
os.environ["HF_HOME"] = "/src/.huggingface/"
os.environ["LPIPS_HOME"] = "/src/models/lpips/"

sys.path.extend([
    "./eden",
    "/clip-interrogator",
    "/lora",
    "/frame-interpolation"
])

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from settings import StableDiffusionSettings
import eden_utils
import film
import interpolator

from cog import BasePredictor, BaseModel, File, Input, Path

film.FILM_MODEL_PATH = "/src/models/film/film_net/Style/saved_model"
interpolator.LPIPS_DIR = "/src/models/lpips/weights/v0.1/alex.pth"

checkpoint_options = [
    #"runwayml/stable-diffusion-v1-5",
    #"prompthero/openjourney-v2",
    "dreamlike-art/dreamlike-photoreal-2.0"
]


class CogOutput(BaseModel):
    file: Path
    name: Optional[str] = None
    thumbnail: Optional[Path] = None
    attributes: Optional[dict] = None
    progress: Optional[float] = None
    isFinal: bool = False


class Predictor(BasePredictor):

    def setup(self):
        print("cog:setup")
        import generation
        generation.CLIP_INTERROGATOR_MODEL_PATH = '/src/cache'

    def predict(
        self,
        
        # Universal args
        mode: str = Input(
            description="Mode", default="generate",
            choices=["generate", "remix", "interpolate", "real2real", "interrogate"]
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
        # ddim_eta: float = 0.0
        # C: int = 4
        # f: int = 8   
        # dynamic_threshold: float = None
        # static_threshold: float = None
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
        # init_image_inpaint_mode: str = Input(
        #     description="Inpainting method for pre-processing init_image when it's masked", 
        #     default="cv2_telea", choices=["mean_fill", "edge_pad", "cv2_telea", "cv2_ns"]
        # ),
        # mask_image_data: str = Input(
        #     description="Load mask image from file, url, or base64 string", 
        #     default=None
        # ),
        # mask_invert: bool = Input(
        #     description="Invert mask", 
        #     default=False
        # ),
        # mask_brightness_adjust: float = 1.0
        # mask_contrast_adjust: float = 1.0

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
        # interpolation_init_images_use_img2txt: bool = Input(
        #     description="Use clip_search to get prompts for the init images, if false use manual interpolation_texts (mode==interpolate)",
        #     default=False
        # ),
        # interpolation_init_images_top_k: int = Input(
        #     description="Top K for interpolation_init_images prompts (mode==interpolate)",
        #     ge=1, le=10, default=2
        # ),
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
        
        # Animation mode
        # animation_mode: str = Input(
        #     description="Interpolation texts (mode==interpolate)",
        #     default='2D', choices=['2D', '3D', 'Video Input']
        # ),
        # init_video: Path = Input(
        #     description="Initial video file (mode==animate)",
        #     default=None
        # ),
        # extract_nth_frame: int = Input(
        #     description="Extract each frame of init_video (mode==animate)",
        #     ge=1, le=10, default=1
        # ),
        # turbo_steps: int = Input(
        #     description="Turbo steps (mode==animate)",
        #     ge=1, le=8, default=3
        # ),
        # previous_frame_strength: float = Input(
        #     description="Strength of previous frame (mode==animate)",
        #     ge=0.0, le=1.0, default=0.65
        # ),
        # previous_frame_noise: float = Input(
        #     description="How much to noise previous frame (mode==animate)",
        #     ge=0.0, le=0.2, default=0.02
        # ),
        # color_coherence: str = Input(
        #     description="Color coherence strategy (mode==animate)", 
        #     default='Match Frame 0 LAB', choices=['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB']
        # ),
        # contrast: float = Input(
        #     description="Contrast (mode==animation)",
        #     ge=0.0, le=2.0, default=1.0
        # ),
        # angle: float = Input(
        #     description="Rotation angle (animation_mode==2D)",
        #     ge=-2.0, le=2.0, default=0.0
        # ),
        # zoom: float = Input(
        #     description="Zoom (animation_mode==2D)",
        #     ge=0.91, le=1.12, default=1.0
        # ),
        # translation_x: float = Input(description="Translation X (animation_mode==3D)", ge=-5, le=5, default=0),
        # translation_y: float = Input(description="Translation U (animation_mode==3D)", ge=-5, le=5, default=0),
        # translation_z: float = Input(description="Translation Z (animation_mode==3D)", ge=-5, le=5, default=0),
        # rotation_x: float = Input(description="Rotation X (animation_mode==3D)", ge=-1, le=1, default=0),
        # rotation_y: float = Input(description="Rotation U (animation_mode==3D)", ge=-1, le=1, default=0),
        # rotation_z: float = Input(description="Rotation Z (animation_mode==3D)", ge=-1, le=1, default=0)

    ) -> Iterator[CogOutput]:

        print("cog:predict:")
        
        import generation
        
        interpolation_texts = interpolation_texts.split('|') if interpolation_texts else None
        interpolation_seeds = [float(i) for i in interpolation_seeds.split('|')] if interpolation_seeds else None
        interpolation_init_images = interpolation_init_images.split('|') if interpolation_init_images else None

        args = StableDiffusionSettings(
            ckpt = random.choice(checkpoint_options),
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
            # init_image_inpaint_mode = init_image_inpaint_mode,
            # mask_image_data = mask_image_data,
            # mask_invert = mask_invert,

            text_input = text_input,
            uc_text = uc_text,
            seed = seed,
            n_samples = n_samples,

            interpolation_texts = interpolation_texts,
            interpolation_seeds = interpolation_seeds,
            interpolation_init_images = interpolation_init_images,
            # interpolation_init_images_use_img2txt = interpolation_init_images_use_img2txt,
            # interpolation_init_images_top_k = interpolation_init_images_top_k,
            interpolation_init_images_power = interpolation_init_images_power,
            interpolation_init_images_min_strength = interpolation_init_images_min_strength,

            n_frames = n_frames,
            scale_modulation = scale_modulation,
            loop = loop,
            smooth = smooth,
            n_film = n_film,
            fps = fps,

            aesthetic_target = None, # None means we'll use the init_images as target
            aesthetic_steps = 10,
            aesthetic_lr = 0.0001,
            ag_L2_normalization_constant = 0.25, # for real2real, only 

            # animation_mode = animation_mode,
            # color_coherence = None if color_coherence=='None' else color_coherence,
            # init_video = init_video,
            # extract_nth_frame = extract_nth_frame,
            # turbo_steps = turbo_steps,
            # previous_frame_strength = previous_frame_strength,
            # previous_frame_noise = previous_frame_noise,
            # contrast = contrast,
            # angle = angle,
            # zoom = zoom,
            # translation = [translation_x, translation_y, translation_z],
            # rotation = [rotation_x, rotation_y, rotation_z]
        )

        print(args)

        out_dir = Path(tempfile.mkdtemp())

        if mode == "interrogate":
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

            out_path = out_dir / f"frame.jpg"
            frame.save(out_path, format='JPEG', subsampling=0, quality=95)
            #progress = s * stream_every / args.steps
            # yield CogOutput(file=out_path, thumbnail=out_path, attributes=None, progress=1.0)
            name = args.text_input
            print(out_path)
            yield CogOutput(file=out_path, name=name, thumbnail=out_path, attributes=attributes, isFinal=True, progress=1.0)
            
        else:
            
            if mode == "interpolate":
                generator = generation.make_interpolation(args)

            elif mode == "real2real":
                args.interpolation_init_images_use_img2txt = True
                generator = generation.make_interpolation(args)

            # elif mode == "animate":
            #     generator = generation.make_animation(args)

            attributes = None
            thumbnail = None

            # generate frames
            for f, (frame, t_raw) in enumerate(generator):
                out_path = out_dir / ("frame_%0.16f.jpg" % t_raw)
                frame.save(out_path, format='JPEG', subsampling=0, quality=95)
                progress = f / args.n_frames
                if not thumbnail:
                    thumbnail = out_path
                #if stream and f % stream_every == 0:
                #    yield CogOutput(file=out_path, thumbnail=out_path, attributes=attributes, progress=progress)

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
