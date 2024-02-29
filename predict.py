# never push DEBUG_MODE = True to Replicate!
DEBUG_MODE = False
#DEBUG_MODE = True

import os
import cv2
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
from PIL import Image
from dataclasses import dataclass, asdict
import numpy as np
import pprint
from cog import BasePredictor, BaseModel, File, Input, Path as cogPath

load_dotenv()

os.environ["TORCH_HOME"] = "/src/.torch"
os.environ["TRANSFORMERS_CACHE"] = "/src/.huggingface/"
os.environ["DIFFUSERS_CACHE"] = "/src/.huggingface/"
os.environ["HF_HOME"] = "/src/.huggingface/"
os.environ["LPIPS_HOME"] = "/src/models/lpips/"

sys.path.extend([
    "./eden",
    "/clip-interrogator",
])

# Eden imports:
from nsfw_detection import lewd_detection
from io_utils import *
from pipe import pipe_manager
from settings import StableDiffusionSettings
import eden_utils
import generation

if DEBUG_MODE:
    debug_output_dir = "/src/tests/server/debug_output"
    if os.path.exists(debug_output_dir):
        shutil.rmtree(debug_output_dir)
    os.makedirs(debug_output_dir, exist_ok=True)

checkpoint_options = [
    "sdxl-v1.0",
    "juggernaut_XL2",
]
checkpoint_default = "juggernaut_XL2"



def extract_n_frames_from_video(video_path, n = 1):
    if n < 1:
        n=1

    save_dir = os.path.dirname(video_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval for frame extraction
    interval = total_frames // n

    # List to store file paths of extracted frames
    extracted_frames_paths = []

    frame_count = 0
    extracted_count = 0

    while cap.isOpened() and extracted_count < n:
        ret, frame = cap.read()
        if ret:
            if frame_count % interval == 0 or (total_frames - frame_count) < interval:
                # Save frame
                frame_path = os.path.join(save_dir, f"frame_{extracted_count}.jpg")
                cv2.imwrite(frame_path, frame)
                extracted_frames_paths.append(frame_path)
                extracted_count += 1
        else:
            break
        frame_count += 1

    cap.release()

    return extracted_frames_paths

class CogOutput(BaseModel):
    files: Optional[list[cogPath]] = []
    name: Optional[str] = None
    thumbnails: Optional[list[cogPath]] = []
    attributes: Optional[dict] = None
    progress: Optional[float] = None
    isFinal: bool = False

class Predictor(BasePredictor):

    GENERATOR_OUTPUT_TYPE = cogPath if DEBUG_MODE else CogOutput

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
            description="Mode", default="create",
            choices=["create", "remix", "upscale", "blend", "controlnet", "interpolate", "real2real", "real2real_audio", "interrogate"]
        ),
        stream: bool = Input(
            description="yield individual results if True", default=False
        ),
        stream_every: int = Input(
            description="for mode create, how many steps per update to stream (steam must be set to True)", 
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
            ge=0.0, le=1.5, default=0.7
        ),
        sampler: str = Input(
            description="Which sampler to use", 
            default="euler", 
            choices=["ddim", "ddpm", "klms", "euler", "euler_ancestral", "dpm", "kdpm2", "kdpm2_ancestral", "pndm"]
        ),
        steps: int = Input(
            description="Diffusion steps", 
            ge=10, le=70, default=35
        ),
        guidance_scale: float = Input(
            description="Strength of text conditioning guidance", 
            ge=0.0, le=20, default=7.5
        ),
        upscale_f: float = Input(
            description="Upscaling resolution",
            ge=1, le=2, default=1
        ),

        # Init image
        init_image: str = Input(
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

        # controlnet image
        controlnet_type: str = Input(
            description="Controlnet type",
            default="off",
            choices=["off", "canny-edge", "depth", "luminance"]
        ),
        control_image: str = Input(
            description="image for controlnet guidance", 
            default=None
        ),
        control_image_strength: float = Input(
            description="Strength of control image", 
            ge=0.0, le=1.5, default=0.0
        ),

        # IP_adapter image
        ip_image: str = Input(
            description="Load ip_adapter image from file, url, or base64 string", 
            default=None
        ),
        ip_image_strength: float = Input(
            description="Strength of image conditioning from ip_adapter (vs txt conditioning from clip-interrogator or prompt) (used in remix, upscale, blend and real2real)", 
            ge=0.0, le=1.25, default=0.65
        ),

        # Create mode
        text_input: str = Input(
            description="Text input", default=None
        ),
        text_inputs_to_interpolate: str = Input(
            description="Text inputs to interpolate, separated by |", default=None
        ),
        text_inputs_to_interpolate_weights: str = Input(
            description="Text input weights to interpolate, separated by |", default=None
        ),
        uc_text: str = Input(
            description="Negative text input (mode==all)",
            default="nude, naked, text, watermark, low-quality, signature, padding, margins, white borders, padded border, moiré pattern, downsampling, aliasing, distorted, blurry, blur, jpeg artifacts, compression artifacts, poorly drawn, low-resolution, bad, grainy, error, bad-contrast"
        ),
        seed: int = Input(
            description="random seed", 
            ge=0, le=1e10, default=13
        ),
        n_samples: int = Input(
            description="batch size",
            ge=1, le=4, default=1
        ),

        # Interpolate mode
        n_frames: int = Input(
            description="Total number of frames for video modes",
            ge=3, le=1000, default=40
        ),
        interpolation_texts: str = Input(
            description="Interpolation texts for video modes",
            default=None
        ),

        interpolation_seeds: str = Input(
            description="Seeds for interpolated texts for video modes",
            default=None
        ),
        interpolation_init_images: str = Input(
            description="Interpolation init images, file paths or urls for video modes",
            default=None
        ),
        interpolation_init_images_power: float = Input(
            description="Power for interpolation_init_images prompts for video modes",
            ge=0.5, le=5.0, default=2.5
        ),
        interpolation_init_images_min_strength: float = Input(
            description="Minimum init image strength for interpolation_init_images prompts for video modes",
            ge=0, le=1.0, default=0.05
        ),
        interpolation_init_images_max_strength: float = Input(
            description="Maximum init image strength for interpolation_init_images prompts for video modes",
            ge=0.0, le=1.0, default=0.95
        ),
        audio_file: str = Input(
            description="An audio file to use for real2real_audio", default=None
        ),
        loop: bool = Input(
            description="Loops (mode==interpolate & real2real)",
            default=True
        ),
        smooth: bool = Input(
            description="Smooth (mode==interpolate & real2real)",
            default=True
        ),
        latent_blending_skip_f: str = Input(
            description="What fraction of the denoising trajectory to skip at the start and end of each interpolation phase, two floats, separated by a pipe (|)",
            default="0.05|0.6"
        ),
        n_anchor_imgs: int = Input(
            description="Number of anchor frames to render (including keyframes) before activating latent blending",
            default=3, ge=3, le=6
        ),
        n_film: int = Input(
            description="Number of times to smooth final frames with FILM (default is 0) (mode==interpolate)",
            default=1, ge=0, le=3
        ),
        fps: int = Input(
            description="Frames per second (mode==interpolate & real2real)",
            default=12, ge=1, le=30
        ),
        use_lcm: bool = Input(
            description="Smooth (mode==interpolate & real2real)",
            default=False
        ),
    ) -> Iterator[GENERATOR_OUTPUT_TYPE]:
    
        for i in range(3):
            print("-------------------------------------------------------")

        print(f"cog:predict: {mode}")
        t_start = time.time()
        
        if init_image == "":
            init_image = None
        if interpolation_init_images == "":
            interpolation_init_images = None

        interpolation_texts = interpolation_texts.split('|') if interpolation_texts else None
        interpolation_seeds = [float(i) for i in interpolation_seeds.split('|')] if interpolation_seeds else None
        interpolation_init_images = interpolation_init_images.split('|') if interpolation_init_images else None

        text_inputs_to_interpolate = text_inputs_to_interpolate.split('|') if text_inputs_to_interpolate else None
        text_inputs_to_interpolate_weights = [float(i) for i in text_inputs_to_interpolate_weights.split('|')] if text_inputs_to_interpolate_weights else None
        assert (text_inputs_to_interpolate is None and text_inputs_to_interpolate_weights is None) or (text_inputs_to_interpolate and text_inputs_to_interpolate_weights), "text_inputs_to_interpolate and text_inputs_to_interpolate_weights must either both be None or both be provided!"
        if text_inputs_to_interpolate and text_inputs_to_interpolate_weights:
            assert len(text_inputs_to_interpolate) == len(text_inputs_to_interpolate_weights), "text_inputs_to_interpolate and text_inputs_to_interpolate_weights must have the same length when provided!"

        lora_path = None
        if lora:
            lora_folder = cogPath('loras')
            lora_zip_path = download(lora, lora_folder)
            lora_path = os.path.join(lora_folder, os.path.splitext(os.path.basename(lora_zip_path))[0])
            extract_to_folder(lora_zip_path, lora_path)

        controlnet_options = {
            "off": None,
            "canny-edge": "controlnet-canny-sdxl-1.0-small",
            "depth":      "controlnet-depth-sdxl-1.0-small",
            "luminance":  "controlnet-luminance-sdxl-1.0",
        }
        
        args = StableDiffusionSettings(
            ckpt = checkpoint,
            lora_path = lora_path,
            lora_scale = lora_scale,

            mode = mode,

            W = width - (width % 8),
            H = height - (height % 8),
            sampler = sampler,
            steps = steps,
            use_lcm = use_lcm,
            guidance_scale = guidance_scale,
            upscale_f = float(upscale_f),

            init_image = init_image,
            init_image_strength = init_image_strength,
            adopt_aspect_from_init_img = adopt_aspect_from_init_img,

            controlnet_path = controlnet_options[controlnet_type],
            control_image = control_image,
            control_image_strength = control_image_strength,
            
            ip_image_strength = ip_image_strength,
            ip_image          = ip_image,

            text_input = text_input,
            uc_text = uc_text,
            seed = seed,
            n_samples = n_samples,

            text_inputs_to_interpolate = text_inputs_to_interpolate,
            text_inputs_to_interpolate_weights = text_inputs_to_interpolate_weights,

            interpolation_texts = interpolation_texts,
            interpolation_seeds = interpolation_seeds,
            interpolation_init_images = interpolation_init_images,
            interpolation_init_images_power = interpolation_init_images_power,
            interpolation_init_images_min_strength = interpolation_init_images_min_strength,
            interpolation_init_images_max_strength = interpolation_init_images_max_strength,
            audio_file = audio_file,

            latent_blending_skip_f = [float(i) for i in latent_blending_skip_f.split('|')],
            n_anchor_imgs = n_anchor_imgs,

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
        out_dir = cogPath(tempfile.mkdtemp())
        
        if DEBUG_MODE:
            lora_str       = f"_lora_{lora_scale}" if lora_path else ""
            controlnet_str = f"_controlnet_{controlnet_type}_{init_image_strength}" if controlnet_type != "off" else ""
            ip_adapter_str = f"_ip_adapter_{ip_image_strength}" if ip_image else ""
            image_str      = f"_image_{init_image_strength:.2f}" if init_image else ""

            prediction_name = f"{int(t_start)}_{mode}{lora_str}{controlnet_str}{ip_adapter_str}{image_str}_upf_{upscale_f:.2f}_{n_frames}_frames"
            base_img_name_no_ext = os.path.basename(control_image).split('.')[0] if control_image else "no_control_image"
            os.makedirs(debug_output_dir, exist_ok=True)

            print(f"DEBUG_MODE: saving to {debug_output_dir}")
            print(f"DEBUG_MODE: prediction_name: {prediction_name}")

            # save a black dummy image to disk so we can easily see which tests failed:
            if mode == "create" or mode == "remix" or mode == "controlnet" or mode == "upscale" or mode == "repaint":
                for index in range(args.n_samples):
                        save_path = os.path.join(debug_output_dir, prediction_name + f"_{index}.jpg")
                        Image.new("RGB", (512, 512), "black").save(save_path)
                        print(f"Saved dummy image to {save_path}")
            else:
                save_path = os.path.join(debug_output_dir, prediction_name + ".jpg")
                Image.new("RGB", (512, 512), "black").save(save_path)


        # throw general user warnings:
        if controlnet_type != "off":
            if args.control_image is None:
                raise ValueError(f"You must provide a shape guidance image when using {controlnet_type} ControlNet!")
            if args.control_image_strength == 0:
                raise ValueError(f"Shape guidance image strength must be > 0.0 when using {controlnet_type} ControlNet!")
        if args.control_image is not None and args.control_image != "":
            if controlnet_type == "off":
                raise ValueError(f"You provided a shape guidance image, but ControlNet type is off!")
            if args.control_image_strength == 0:
                raise ValueError(f"Shape guidance image strength must be > 0.0 when using {controlnet_type} ControlNet!")

        print("------------- args: -------------")
        pprint.pprint(asdict(args), indent=4)
        print("---------------------------------")

        if mode == "interrogate":
            interrogation = generation.interrogate(args)
            out_path = out_dir / f"interrogation.txt"
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(interrogation)
            attributes = {'interrogation': interrogation}
            attributes['job_time_seconds'] = time.time() - t_start
            
            if DEBUG_MODE:
                shutil.copyfile(out_path, os.path.join(debug_output_dir, "out_interrogation.txt"))
                yield out_path
            else:
                yield CogOutput(files=[out_path], name=interrogation, thumbnails=[out_path], attributes=attributes, isFinal=True, progress=1.0)
        
        elif mode == "create" or mode == "remix" or mode == "controlnet" or mode == "upscale" or mode == "repaint":
            
            if (mode == "upscale" or mode == "remix") and (not args.init_image):
                raise ValueError(f"an init_image must be provided for mode = {mode}")

            if args.controlnet_path:
                if not args.control_image:
                    raise ValueError(f"a control_image must be provided when using controlnet")
                if args.control_image_strength == 0.0:
                    raise ValueError("controlnet requires init_image_strength > 0.0")
                
            if args.init_image is None:
                args.init_image_strength = 0.0

            attributes = {}
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

            # Run nsfw-detection:
            attributes['nsfw_scores'] = lewd_detection(out_paths)
            attributes['job_time_seconds'] = time.time() - t_start

            if not batch_i_args.name:
                batch_i_args.name = "Eden creation" #Make sure we always have a non-emtpy name

            if DEBUG_MODE:
                print(attributes)
                for index, out_path in enumerate(out_paths):
                    print(f'Orig image: {out_path}')
                    print(f"Copying image to {debug_output_dir}..")
                    shutil.copyfile(out_path, os.path.join(debug_output_dir, prediction_name + f"_{index}.jpg"))
                yield out_paths[0]
            else:
                yield CogOutput(files=out_paths, name=batch_i_args.name, thumbnails=out_paths, attributes=attributes, isFinal=True, progress=1.0)

        elif mode == "real2real_audio":
            if not args.audio_file:
                raise ValueError("You must provide an audio file to use real2real_audio")
            if len(args.interpolation_init_images) < 2:
                raise ValueError("You must provide at least 2 images to interpolate!")
            if args.interpolation_seeds:
                if len(args.interpolation_init_images) != len(args.interpolation_seeds):
                 raise ValueError("You must provide the same amount of seeds as images!")

            audio_folder = cogPath('audio_files')
            args.audio_file = download(args.audio_file, audio_folder)
            args.name = "Audio reactive real2real"

            # notify UI we started running!
            yield CogOutput(attributes={}, progress=0.05)

            t_start = time.time()
            out_path = generation.real2real_audioreactive(args.interpolation_init_images, args.audio_file,
                args.lora_path, 
                render_settings = {
                    "W": args.W,
                    "H": args.H,
                    "steps": args.steps,
                    "seconds_between_keyframes": 12, # determines the visual density of the video
                    "fps": args.fps},
                audio_reactivity_settings = {
                        'depth_rescale'     : [105., 255.],
                        '3d_motion_xyz'     : [0.7, 0.7, -90],
                        'circular_motion_period_s': 15,  # the period of the circular xy motion around the center (in seconds)
                        '3d_rotation_xyz'   : [0,0,0],
                        'brightness_factor' : 0.0005, # 0.001
                        'contrast_factor'   : 0.2, #0.4
                        'saturation_factor' : 0.25, #0.5
                        '2d_zoom_factor'    : 0.00,
                        'noise_factor'      : 0.0},
                seed=args.seed, 
                interpolation_seeds=args.interpolation_seeds, 
                loop=args.loop,
                save_distance_data = True,
                )

            yield CogOutput(attributes={}, progress=1.0)
            
            video_frame_paths = extract_n_frames_from_video(out_path, n = len(args.interpolation_init_images))

            attributes = {}
            frame_nsfw_scores = lewd_detection(video_frame_paths)
            attributes['nsfw_scores']      = [np.max(frame_nsfw_scores)]
            attributes['job_time_seconds'] = time.time() - t_start

            if DEBUG_MODE:
                print(attributes)
                shutil.copyfile(out_path, os.path.join(debug_output_dir, prediction_name + ".mp4"))
                yield out_path
            else:
                yield CogOutput(files=[cogPath(out_path)], name=args.name, thumbnails=[cogPath(video_frame_paths[0])], attributes=attributes, isFinal=True, progress=1.0)

        else: # mode == "interpolate" or mode == "real2real" or mode == "blend" or mode == "real2real_audio"

            if args.controlnet_path and args.control_image_strength == 0.0:
                raise ValueError("controlnet requires init_image_strength > 0.0")

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
            if (mode == "interpolate" and len(args.interpolation_texts) < 2):
                raise ValueError("Must have at least two prompts to interpolate!")
                
            if (mode == "real2real" and len(args.interpolation_init_images) < 2):
                raise ValueError("Must have at least two init_images to do real2real!")

            loop = (args.loop and len(args.interpolation_seeds) == 2)

            if loop:
                args.n_frames = args.n_frames // 2
            
            generator = generation.make_interpolation(args, force_timepoints=force_timepoints)
            attributes = {}
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

                if not DEBUG_MODE:
                    progress = f / args.n_frames
                    cog_output = CogOutput(attributes=attributes, progress=progress)
                    if stream and f % stream_every == 0:
                        cog_output.files = [out_path]

                    yield cog_output
                else:
                    # make a subdir for the frames:
                    frames_dir = os.path.join(debug_output_dir, f"{prediction_name}_frames")
                    os.makedirs(frames_dir, exist_ok=True)
                    shutil.copyfile(out_path, os.path.join(frames_dir, f"frame_{t_raw:0.16f}.jpg"))

            # run FILM
            if args.n_film > 0:
                try:
                    if args.W * args.H > 1600*1600:
                        print("Clearing SD pipe memory to run FILM...")
                        pipe_manager.clear()

                    print('predict.py: running FILM...')
                    FILM_MODEL_PATH = "/src/models/film/film_net/Style/saved_model"
                    abs_out_dir_path = os.path.abspath(str(out_dir))
                    command = ["python", "/src/eden/film.py", "--frames_dir", abs_out_dir_path, "--times_to_interpolate", str(args.n_film), '--update_film_model_path', FILM_MODEL_PATH]
                    
                    run_and_kill_cmd(command)
                    print("predict.py: FILM done.")
                    film_out_dir = cogPath(os.path.join(abs_out_dir_path, "interpolated_frames"))

                    # check if film_out_dir exists and contains at least 3 .jpg files:
                    if os.path.exists(film_out_dir) and len(list(film_out_dir.glob("*.jpg"))) > 3:
                        out_dir = film_out_dir
                    else:
                        print("ERROR: film_out_dir does not exist or contains less than 3 .jpg files, using original frames instead.")
                except Exception as e:
                    print(str(e))
                    print("Something went wrong with FILM, using original frames instead.")

            if mode != "blend":
                # save video
                out_path = cogPath(out_dir) / "out.mp4"
                eden_utils.write_video(out_dir, str(out_path), loop=loop, fps=args.fps)

                if mode == "real2real":
                    attributes = {"interrogation": args.interpolation_texts}

            # run NSFW detection on thumbnail:
            attributes['nsfw_scores'] = lewd_detection([str(thumbnail)])
            attributes['job_time_seconds'] = time.time() - t_start
            
            if not args.name:
                args.name = "Eden creation" #Make sure we always have a non-emtpy name

            if DEBUG_MODE:
                print(attributes)
                if mode == "blend":
                    shutil.copyfile(out_path, os.path.join(debug_output_dir, prediction_name + ".jpg"))
                else:
                    shutil.copyfile(out_path, os.path.join(debug_output_dir, prediction_name + ".mp4"))
                yield out_path
            else:
                yield CogOutput(files=[out_path], name=args.name, thumbnails=[thumbnail], attributes=attributes, isFinal=True, progress=1.0)

        if DEBUG_MODE:
            print("--------------------------------")
            print("--- cog was in DEBUG mode!!! ---")
            print("--------------------------------")

        t_end = time.time()
        print(f"predict.py: done in {t_end - t_start:.2f} seconds")
