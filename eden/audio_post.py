import sys, os, shutil
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image, ImageOps

import settings
from settings import StableDiffusionSettings
from generation import *
from eden_utils import *
from planner import Planner

from depth.depth_transforms import *

def predict_depth_map_zoe(pil_image, zoe, depth_rescale, flip_aug = False):
    min_v, max_v = depth_rescale

    depth_tensor = zoe.infer_pil(pil_image, output_type="tensor")
    if flip_aug:
        flipped_tensor = zoe.infer_pil(ImageOps.mirror(pil_image), output_type="tensor")
        depth_tensor = 0.5 * (depth_tensor + torch.flip(flipped_tensor, dims=[1]))

    # renormalize depth map:
    depth_tensor  = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min()) * (max_v - min_v) + min_v
    pil_depth_map = Image.fromarray(depth_tensor.permute(0, 1).cpu().numpy().astype(np.uint8))

    return pil_depth_map, depth_tensor

def predict_depth_map_midas(pil_image, depth_estimator, feature_extractor, depth_rescale):
    min_v, max_v = depth_rescale

    width, height = pil_image.size
    image = feature_extractor(images=pil_image, return_tensors="pt").pixel_values.to("cuda")

    depth_tensor = depth_estimator(image).predicted_depth

    depth_tensor = torch.nn.functional.interpolate(
        depth_tensor.unsqueeze(1),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )
    depth_tensor = 1 - (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min())
    depth_tensor = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min()) * (max_v - min_v) + min_v

    pil_depth_map = Image.fromarray((depth_tensor.permute(0, 2, 3, 1).cpu().numpy()[0].squeeze()).astype(np.uint8))

    return pil_depth_map, depth_tensor


def depth_warp(frames_dir, audio_planner, audio_reactivity_settings, 
    depth_model = "zoe",
    save_depth_maps = 0):

    translate_xyz = audio_reactivity_settings['3d_motion_xyz']
    rotate_xyz =  audio_reactivity_settings['3d_rotation_xyz']

    anim_args = AnimArgs(
        near_plane=random.choice([200.0]),
        far_plane=random.choice([20000]),
        fov=random.choice([40]),
        sampling_mode="bicubic",
        padding_mode="reflection"
    )

    #warp_name = f"warp_trans_{translate_xyz[0]}_{translate_xyz[1]}_{translate_xyz[2]}_rot_{rotate_xyz[0]}_{rotate_xyz[1]}_{rotate_xyz[2]}_fov_{anim_args.fov}_near_{anim_args.near_plane}_far_{anim_args.far_plane}_minv_{min_v}"
    warp_name = "_warped"
    output_dir = os.path.join(frames_dir, "depth_warped")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if save_depth_maps:
        depth_dir = os.path.join(frames_dir, "depth_maps")
        os.makedirs(depth_dir, exist_ok=True)

    torch.cuda.empty_cache()
    if depth_model == "midas":
        depth_estimator   = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
        feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
    else:
        repo = "isl-org/ZoeDepth"
        #torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True) 
        model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True).to(settings._device)

    frame_paths = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg") and "_depth" not in f])
    print("Found %d frames!" % len(frame_paths))

    with torch.no_grad():
        print("Predicting depth maps...")
        for i, frame_path in tqdm(enumerate(frame_paths)):
            translate_xyz_frame = translate_xyz.copy()

            frame = Image.open(frame_path)
            if depth_model == "midas":
                depth_map, depth_tensor = predict_depth_map_midas(frame, depth_estimator, feature_extractor, audio_reactivity_settings['depth_rescale'])
            else:
                depth_map, depth_tensor = predict_depth_map_zoe(frame, model_zoe_nk, audio_reactivity_settings['depth_rescale'])

            if save_depth_maps:
                depth_map.save(os.path.join(depth_dir, os.path.basename(frame_path)), quality=95)

            # apply a 3d depth warp to the frame using the depth map:
            warp_factor = audio_planner.fps_adjusted_percus_features[0, i]
            warp_factor = np.clip(warp_factor, 0.0, 1.0)

            # make x,y rotate in a circle with amplitude A and period P:
            P = audio_reactivity_settings['circular_motion_period_s'] * audio_planner.fps
            translate_xyz_frame[0] = rotate_xyz[0] * np.sin(2*np.pi*i/P)
            translate_xyz_frame[1] = rotate_xyz[1] * np.cos(2*np.pi*i/P)

            # apply the warp_factor to translate_xyz and rotate_xyz:
            translate_xyz_frame[2] *= warp_factor
            rotate_xyz_frame = [warp_factor * r for r in rotate_xyz]
            warped_frame = anim_frame_warp_3d(np.array(frame), depth_tensor, anim_args, translate_xyz_frame, rotate_xyz_frame)
            warped_frame = Image.fromarray(warped_frame)

            # save warped frame:
            save_name = os.path.basename(frame_path).replace(".jpg", f"{warp_name}_depth.jpg")
            warped_frame.save(os.path.join(output_dir, save_name), quality=95)

            if 0:
                # save orig frame to the same dir:
                orig_img_path = os.path.join(output_dir, os.path.basename(frame_path))
                if not os.path.exists(orig_img_path):
                    frame.save(orig_img_path, quality=95)

    return output_dir


def make_audio_reactive(frames_dir, audio_planner, audio_reactivity_settings):
    outdir = os.path.join(frames_dir, "audio_reactive_frames")
    os.makedirs(outdir, exist_ok=True)
    frame_paths = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    print("Found %d frames!" % len(frame_paths))

    for i, frame_path in tqdm(enumerate(frame_paths)):
        frame = audio_planner.morph_image(Image.open(frame_path), frame_index = i, audio_reactivity_settings = audio_reactivity_settings)
        frame.save(os.path.join(outdir, os.path.basename(frame_path)), quality=95)
    
    return outdir

def post_process_audio_reactive_video_frames(frames_dir, audio_path, fps, n_film, 
    update_audio_reactivity_settings = None):
    
    audio_reactivity_settings = {
                'depth_rescale'     : [105., 255.],
                '3d_motion_xyz'     : [0.7, 0.7, -90],
                'circular_motion_period_s': 15,  # the period of the circular xy motion around the center (in seconds)
                '3d_rotation_xyz'   : [0,0,0],
                'brightness_factor' : 0.003,
                'contrast_factor'   : 0.4,
                'saturation_factor' : 0.5,
                '2d_zoom_factor'    : 0.00,
                'noise_factor'      : 0.0,
    }

    if update_audio_reactivity_settings is not None:
        audio_reactivity_settings.update(update_audio_reactivity_settings)

    output_video_dir = os.path.dirname(frames_dir)
    name_str = os.path.basename(frames_dir) + "_post"

    if audio_path is not None:
        frame_paths = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        audio_planner = Planner(audio_path, fps, len(frame_paths))
        frames_dir = depth_warp(frames_dir, audio_planner, audio_reactivity_settings)

    return

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
        audio_planner = Planner(audio_path, fps, len(frame_paths))
        frames_dir = make_audio_reactive(frames_dir, audio_planner, audio_reactivity_settings)

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

    return fin_video_path




################################################################################################



if __name__ == "__main__":

    audio_path = ("/data/xander/Projects/cog/stable-diffusion-dev/eden/xander/tmp_unzip/features.pkl", "/data/xander/Projects/cog/stable-diffusion-dev/eden/xander/tmp_unzip/music.mp3")
    
    fps = 12    # orig fps, before FILM
    n_film = 1  # set n_film to 0 to disable FILM interpolation

    frames_dir = "/data/xander/Projects/cog/eden-sd-pipelines/eden/templates/results_real2real_audioreactive_demo"

    if 0:
        # grab all the subdirectories of frames_dir:
        frames_dirs = [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, f))]
        for frames_dir in sorted(frames_dirs):
            print(f"Postprocessing {frames_dir}...")
            post_process_audio_reactive_video_frames(frames_dir, audio_path, fps, n_film)
    else:
        frames_dir = "/data/xander/Projects/cog/eden-sd-pipelines/eden/templates/results_test"
        for i in range(1):
            post_process_audio_reactive_video_frames(frames_dir, audio_path, fps, n_film)

















###################################### DEPRECATED ######################################

def get_fps(input_video_path):
    cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=s=x:p=0 {input_video_path}"
    fps = os.popen(cmd).read().strip()
    fps = float(fps.split("/")[0]) / float(fps.split("/")[1])
    print("Input video fps:", fps)
    return fps


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