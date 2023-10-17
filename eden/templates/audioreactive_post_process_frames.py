import sys, os
import numpy as np
import torch
sys.path.append('..')

from settings import StableDiffusionSettings
from generation import *
from eden_utils import *

def adjust_n_steps(steps, init_image_strength, min_max_steps):
    if steps * (1-init_image_strength) > min_max_steps[1]:
        return int(min_max_steps[1] / (1-init_image_strength))
    elif steps * (1-init_image_strength) < min_max_steps[0]:
        return int(min_max_steps[0] / (1-init_image_strength))
    else:
        return steps

def compute_target_resolution(W,H,total_pixels):
    # Determine the current target resolution based on total_pixels:
    aspect_ratio = W / H

    # Compute the target resolution:
    W, H = np.sqrt(total_pixels) * np.sqrt(aspect_ratio), np.sqrt(total_pixels) / np.sqrt(aspect_ratio)

    # Round W and H to the nearest multiple of 64:
    W, H = int(np.round(W / 64) * 64), int(np.round(H / 64) * 64)
    return W, H

def get_frames_and_metadata(input_frames_dir):
    """
    Returns a list of frames and a dict of metadata
    """

    metadata_dir = os.path.join(input_frames_dir, "phase_data")
    all_files    = os.listdir(input_frames_dir)

    # check if the metadata folder exists:
    assert os.path.exists(metadata_dir), f"Could not find phase_data folder in {input_frames_dir}"

    phase_data_paths = sorted([os.path.join(metadata_dir, f) for f in os.listdir(metadata_dir) if f.endswith(".npz")])
    frame_paths      = sorted([os.path.join(input_frames_dir, f) for f in all_files if f.endswith(".jpg")])
    args             = json.load(open(os.path.join(metadata_dir, "args.json")))

    print(f"Found {len(frame_paths)} frames and {len(phase_data_paths)} phase_data files in {input_frames_dir}")
    return frame_paths, phase_data_paths, args


def get_conditioning(framepath, phase_data_paths):
    # Get the frame's t_raw:
    t_raw = float(framepath.split("frame_")[1].split("_")[0])

    # Get the corresponding phase_data file:
    for phase_data_path in sorted(phase_data_paths):
        name_str = os.path.basename(phase_data_path).split(".npz")[0]
        t_min, t_max = name_str.split("_to_")
        if (float(t_min) <= t_raw) and (t_raw < float(t_max)):
            break

    phase_data = np.load(phase_data_path, allow_pickle=True)
    phase_data = {key: phase_data[key] for key in phase_data.files}

    # Find the index of t_raw in t_raws:
    t_raws = phase_data["t_raw"]
    t_raw_idx = np.argmin(np.abs(t_raws - t_raw))

    if np.abs(t_raws[t_raw_idx] - t_raw) > 0.01:
        print(f"Using metadata from frame with t_raw: {t_raws[t_raw_idx]:.4f} for t_raw: {t_raw:.4f}")

    uc_vector = phase_data["uc"].astype(np.float32)
    c_vector  = phase_data["c"][t_raw_idx].astype(np.float32)
    scale     = phase_data["scale"][t_raw_idx]

    # convert vectors to torch tensors:
    uc_vector = torch.from_numpy(uc_vector).to(device)
    c_vector  = torch.from_numpy(c_vector).to(device)

    return uc_vector, c_vector, scale


def upscale_frame_directory(input_dir, 
    total_pixels = (896)**2,     # higher values will require more GPU memory
    init_image_strength = 0.6,
    target_steps = 80, min_max_steps = [15,20],
    seed = 0,
    force_sampler = None):

    """
    Upscale a directory of SD generated .jpg images (with corresponding .json configs) to a higher resolution.
    """

    outdir = os.path.join(input_dir, "upscaled")
    os.makedirs(outdir, exist_ok=True)
    frame_paths, phase_data_paths, _args = get_frames_and_metadata(input_dir)

    args = StableDiffusionSettings()
    # copy over all possible values from _args to args:
    for key in _args:
        if key in args.__dict__:
            setattr(args, key, _args[key])

    # overwrite upscaling args:
    args.init_image_strength = init_image_strength
    args.steps = adjust_n_steps(target_steps, init_image_strength, min_max_steps)
    args.W, args.H = compute_target_resolution(args.W, args.H, total_pixels)
    args.seed = seed
    args.interpolator = None
    if force_sampler is not None:
        args.sampler = force_sampler

    for i, frame_path in enumerate(frame_paths):
        args.init_latent = None
        args.init_image = load_img(frame_path, 'RGB')
        args.uc, args.c, args.scale = get_conditioning(frame_path, phase_data_paths)

        filename = f"{os.path.basename(frame_path)}_HD_{init_image_strength:.2f}_{args.sampler}"
        outfilepath = os.path.join(outdir, filename+".jpg")
        if os.path.exists(outfilepath):
            print(f"Skipping {filename}..")
            continue

        print(f"Upscaling frame {i+1}/{len(frame_paths)}: {os.path.basename(frame_path)}")
        _, new_images = generate(args)
        frame = Image.fromarray(new_images[0].cpu().numpy().astype(np.uint8))
        frame.save(outfilepath, quality=95)

    return outdir

def get_fps(input_video_path):
    cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=s=x:p=0 {input_video_path}"
    fps = os.popen(cmd).read().strip()
    fps = float(fps.split("/")[0]) / float(fps.split("/")[1])
    print("Input video fps:", fps)
    return fps

def extract_frames(video_path, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    cmd = f"ffmpeg -i {video_path} -qscale:v 0 {frames_dir}/%07d.jpg"
    print(cmd)
    os.system(cmd)
    return frames_dir

def smooth(input_path, n_film, out_video_path, sigma_seconds, video_crf = 23, print_f = 20):

    """
    
    Applies FILM smoothing + Latent space smoothing to the frames of a video
    input_path can be a .mp4 file or a directory of frames

    """
    
    tmp_dir = "tmp"
    frames_dir = os.path.join(tmp_dir, "tmp_frames")
    outdir     =  os.path.join(tmp_dir, "tmp_frames_smoothed")

    if input_path.endswith(".mp4"):
        # get fps of input video:
        fps = get_fps(input_path)
        # Extract frames from video
        frames_dir = extract_frames(input_path, frames_dir)
    else:
        # Assume input_path is a directory of frames
        frames_dir = input_path
        fps = 10

    if n_film > 0:
        # Run FILM postprocessing (frame blending)
        frames_dir = interpolate_FILM(frames_dir, n_film)

    # Now, iteratively load all the frames and encode them into VQGAN latent space:
    args = StableDiffusionSettings()
    model = get_model(args.config, args.ckpt, args.half_precision)

    latents = []
    frame_paths = sorted(os.listdir(frames_dir))
    for i, frame_filename in enumerate(frame_paths):
        if frame_filename.endswith(".jpg"):
            frame_path = os.path.join(frames_dir, frame_filename)
            if i%print_f==0:
                print(f"Encoding frame {i:04d}/{len(frame_paths):04d}...")
            frame = load_img(frame_path, 'RGB')
            args.W, args.H = frame.size
            frame = preprocess_image(frame, shape=(args.W, args.H)).to(device)
            latent = model.get_first_stage_encoding(model.encode_first_stage(frame))
            latents.append(latent)

    latents = torch.cat(latents, dim=0)


    out_video_fps = fps*n_film if n_film != 0 else fps
    sigma = sigma_seconds * out_video_fps
    ts = np.arange(0, len(latents))

    # Now, decode the latents back into images:
    os.makedirs(outdir, exist_ok=True)
    for i, latent in enumerate(latents):
        if i%print_f==0:                
            print(f"Decoding smoothed frame {i:04d}/{len(latents):04d}..")

        kernel_weights = np.array([np.exp(-((ts[i] - ts[j]) ** 2) / (2 * sigma ** 2)) for j in range(len(ts))])
        kernel_weights /= np.sum(kernel_weights)
        kernel_weights = torch.from_numpy(kernel_weights).to(device)

        # Add dummy dimensions to the weights to match the latents:
        kernel_weights = kernel_weights.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        # Smooth the latent:
        latent = torch.sum(latents * kernel_weights, dim=0)
        latent = latent.half() if args.half_precision else latent.float()

        # Decode the latent:
        x_samples = model.decode_first_stage(latent.unsqueeze(0), force_not_quantize=True)[0]
        frame = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        frame = 255. * rearrange(frame, 'c h w -> h w c')
        frame = Image.fromarray(frame.cpu().numpy().astype(np.uint8))
        frame.save(os.path.join(outdir, "%07d.jpg" % i), quality=95)

    # Finally, encode the frames into a video:
    cmd = f"ffmpeg -framerate {out_video_fps} -pattern_type glob -i '{outdir}/*.jpg' -vcodec libx264 -crf {video_crf} -pix_fmt yuv420p {out_video_path}"
    os.system(cmd)

    # Cleanup
    os.system(f"rm -rf {tmp_dir}")
    print(f"All done, smoothed video is in {out_video_path}")

def make_audio_reactive(frames_dir, audio_planner):
    paths = os.listdir(frames_dir)
    if "interpolated_frames" in paths:
        frames_dir = os.path.join(frames_dir, "interpolated_frames")

    outdir = os.path.join(frames_dir, "audio_reactive_frames")
    os.makedirs(outdir, exist_ok=True)
    frame_paths = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    print("Found %d frames!" % len(frame_paths))

    for i, frame_path in enumerate(frame_paths):
        frame = audio_planner.morph_image(Image.open(frame_path), frame_index = i)
        frame.save(os.path.join(outdir, os.path.basename(frame_path)), quality=95)

        if i%20==0:
            print(f"Completed frame {i+1}/{len(frame_paths)}")
    
    return outdir

def post_process_audio_reactive_video_frames(frames_dir, audio_path, fps, n_film):

    output_video_dir = os.path.dirname(frames_dir)
    name_str = os.path.basename(frames_dir) + "_post"

    if 0: # upscale the frames, currently disabled
        frames_dir = upscale_frame_directory(frames_dir, 
            total_pixels = (1024+264)**2,     # higher values will require more GPU memory
            init_image_strength = 0.66,
            target_steps = 100, min_max_steps = [25,40])

    if n_film > 0:
        frames_dir = os.path.abspath(frames_dir)

        try:
            command = [sys.executable, os.path.join(str(SD_PATH), "eden/film.py"), "--frames_dir", frames_dir, "--times_to_interpolate", str(n_film)]

            print("running command:", ' '.join(command))
            result = subprocess.run(command, text=True, capture_output=True)
            print(result)
            print(result.stdout)

            film_out_dir = Path(os.path.join(frames_dir, "interpolated_frames"))

            # check if film_out_dir exists and contains at least 3 .jpg files:
            if os.path.exists(film_out_dir) and len(list(film_out_dir.glob("*.jpg"))) > 3:
                frames_dir = str(film_out_dir)
                fps = fps*(1+n_film)
            else:
                print("ERROR: film_out_dir does not exist or contains less than 3 .jpg files, using original frames instead.")
            
        except Exception as e:
            print(str(e))
            print("Something went wrong with FILM, using original frames instead.")

    frame_paths = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])

    if audio_path is not None:
        from planner import Planner
        audio_planner = Planner(audio_path, fps, len(frame_paths))
        frames_dir = make_audio_reactive(frames_dir, audio_planner)

    if 0: # deprecated, TODO: remove this?
        n_film     = 0
        video_crf  = 18
        sigma_seconds = 0.05  # sigma of the latent smoothing kernel in seconds
        out_video_path = os.path.splitext(input_path)[0] + f"_smoothed_n_film_{n_film}.mp4"
        #smooth(input_path, n_film, out_video_path, sigma_seconds, video_crf = video_crf)

    video_path = os.path.join(os.path.dirname(frames_dir), f"{name_str}.mp4")
    write_video(frames_dir, video_path, fps=fps)

    if audio_path is not None:
        print("adding audio...")
        fin_video_path = video_path.replace(".mp4", "_audio.mp4")
        add_audio_to_video(audio_planner.audio_path, video_path, fin_video_path)
    else:
        fin_video_path = video_path

    os.system(f"mv {fin_video_path} {os.path.join(output_video_dir, os.path.basename(fin_video_path))}")
    print(f"final video is at {fin_video_path}")

"""

export CUDA_VISIBLE_DEVICES=0
cd /data/xander/Projects/cog/eden-sd-pipelines/eden/templates
python audioreactive_post_process_frames.py

"""

if __name__ == "__main__":

    audio_path = ("/data/xander/Projects/cog/stable-diffusion-dev/eden/xander/tmp_unzip/features.pkl", "/data/xander/Projects/cog/stable-diffusion-dev/eden/xander/tmp_unzip/music.mp3")
    
    fps = 12    # orig fps, before FILM
    n_film = 1  # set n_film to 0 to disable FILM interpolation

    root_dir = "/data/xander/Projects/cog/eden-sd-pipelines/eden/templates/results_real2real_audioreactive_test"
    subdirs = os.listdir(root_dir)

    for subdir in subdirs:
        frames_dir = os.path.join(root_dir, subdir)

        if not os.path.isdir(frames_dir):
            continue
            
        #frames_dir = os.path.join(frames_dir, "interpolated_frames")
        post_process_audio_reactive_video_frames(frames_dir, audio_path, fps, n_film)
