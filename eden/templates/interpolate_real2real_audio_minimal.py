import os, random, sys, tempfile
sys.path.append('..')
from settings import StableDiffusionSettings
from generation import *
from templates.interpolate_real2real import real2real
from planner import Planner
from audio_post import post_process_audio_reactive_video_frames
from extract_audio_features_eden import extract_audio_features

def get_random_img_paths_from_dir(directory_path, n_imgs, sorted = False):
    img_exts = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]
    all_img_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
    all_img_paths = [f for f in all_img_paths if os.path.splitext(f)[1] in img_exts]

    paths = random.sample(all_img_paths, n_imgs)

    if sorted: 
        paths.sort()

    return paths


def real2real_audioreactive(
        input_images, audio_path, 
        render_settings = {
            "W": 640,
            "H": 640,
            "n_steps": 25,
            "seconds_between_keyframes": 12,
            "fps": 9,
        },
        audio_reactivity_settings = {
                'depth_rescale'     : [105., 255.],
                '3d_motion_xyz'     : [0.7, 0.7, -90],
                'circular_motion_period_s': 15,  # the period of the circular xy motion around the center (in seconds)
                '3d_rotation_xyz'   : [0,0,0],
                'brightness_factor' : 0.0005, # 0.001
                'contrast_factor'   : 0.4, #0.4
                'saturation_factor' : 0.5, #0.5
                '2d_zoom_factor'    : 0.00,
                'noise_factor'      : 0.0},
        do_post_process = True,
        output_dir = None,
        save_distance_data = True, # save perceptual distance plots to disk
        loop = True, # wrap the video to make it loop-able
        seed = 0,
        interpolation_seeds = None,
        debug = False
        ):

    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    n = len(input_images)

    if debug: # debug: fast render settings
        render_settings["H"]       = 640
        render_settings["W"]       = 640
        render_settings['n_steps'] = 20
        n = 4
        input_images = input_images[:n]
        render_settings["seconds_between_keyframes"] = 5

    # Compute number of frames between keyframes:
    inter_frames = int(render_settings["seconds_between_keyframes"] * render_settings["fps"])

    args = StableDiffusionSettings(
        H = render_settings["H"],
        W = render_settings["W"],
        steps = render_settings['n_steps'],
        n_frames = inter_frames*(n-1) + n,
        guidance_scale = 8,
        text_input = "real2real",  # text_input is also the title, but has no effect on interpolations
        interpolation_seeds = [random.randint(1, 1e8) for _ in range(n)] if (not interpolation_seeds) else interpolation_seeds,
        interpolation_init_images = input_images,
        ip_image_strength = 1.0,
        n_anchor_imgs = 3,
        latent_blending_skip_f = [0.07, 0.8],
        loop = loop,
        fps = render_settings["fps"],
        seed = seed,
    )

    if args.loop:
        args.n_frames = inter_frames*((n+1)-1) + (n+1)

    audio_path   = extract_audio_features(audio_path, re_encode = 1)
    args.planner = Planner(audio_path, args.fps, args.n_frames)

    video_path, frames_dir, timepoints = real2real(input_images, output_dir, 
                name_str = f"seed_{args.seed}", args = args, seed = args.seed, remove_frames_dir = False, 
                save_video = 1, save_phase_data=False, save_distance_data = save_distance_data)
    
    if do_post_process: # apply depth warping and audio-reactive saturation/contrast adjustments
        n_film = 1
        fin_video_path = post_process_audio_reactive_video_frames(frames_dir, audio_path, args.fps, n_film, audio_reactivity_settings)
    else:
        print("adding audio...")
        fin_video_path = video_path.replace(".mp4", "_audio.mp4")
        add_audio_to_video(audio_path[1], video_path, fin_video_path)

    return fin_video_path


if __name__ == "__main__":

    # main render settings
    audio_path = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/audio/versilov/audio.mp3"
    n_imgs     = 6
    seed       = 1
    
    # Get random images from a directory:
    input_dir  = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/init_imgs/flow"
    output_dir = 'results_real2real_audioreactive_club_long'

    seed_everything(seed)
    img_paths = get_random_img_paths_from_dir(input_dir, n_imgs, sorted=False)
    fin_video_path = real2real_audioreactive(img_paths, audio_path, output_dir = output_dir, seed=seed)