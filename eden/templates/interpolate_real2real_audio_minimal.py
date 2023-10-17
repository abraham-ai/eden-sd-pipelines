import os, random, sys
sys.path.append('..')
from settings import StableDiffusionSettings
from generation import *
from templates.interpolate_real2real import real2real
from planner import Planner


def real2real_x(W, H, args, input_images, outdir, n, exp_name = "", audio_path = None, save_phase_data = False, save_distance_data = False):

    name_str = f"seed_{args.seed}_{exp_name}"

    if audio_path is not None:
        n_final_frames = args.n_frames
        args.planner = Planner(audio_path, args.fps, n_final_frames)

    video_path, frames_dir, timepoints = real2real(input_images, outdir, 
                name_str = name_str, args = args, seed = args.seed, remove_frames_dir = False, 
                save_video = 1, save_phase_data=save_phase_data, save_distance_data = save_distance_data)
    
    if audio_path is not None:
        print("adding audio...")
        add_audio_to_video(args.planner.audio_path, video_path, video_path.replace(".mp4", "_audio.mp4"))

    return frames_dir


"""

conda activate diffusers
cd /home/rednax/SSD2TB/Github_repos/cog/eden-sd-pipelines/eden/templates
python interpolate_real2real_audio_minimal.py

"""

def get_random_img_paths_from_dir(directory_path, n_imgs):
    img_exts = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]
    all_img_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
    all_img_paths = [f for f in all_img_paths if os.path.splitext(f)[1] in img_exts]
    return random.sample(all_img_paths, n_imgs)

from audioreactive_post_process_frames import post_process_audio_reactive_video_frames

if __name__ == "__main__":

    ##############################################################################
    
    # main render settings (eg specified by the user)
    H,W          = 1024+256, 1024+256
    n_steps      = 50       # n_diffusion steps per frame
    output_fps   = 12      
    seconds_between_keyframes = 7
    inter_frames = int(seconds_between_keyframes * output_fps)

    # audio_path is either a path of a .zip file, or a tuple of (audio_features_pickle, audio_mp3_path)
    audio_path = ("path_to_features.pkl", "path_to_audio.mp3")
    audio_path = ("/data/xander/Projects/cog/stable-diffusion-dev/eden/xander/tmp_unzip/features.pkl", "/data/xander/Projects/cog/stable-diffusion-dev/eden/xander/tmp_unzip/music.mp3")
    
    # Get random images from a directory: (this should be replaced with timeline imgs in WZRD)
    n_imgs = 8
    input_dir = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/init_imgs/diverse_real2real_seeds"
    #input_dir = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/init_imgs/tall"

    outdir    = 'results_real2real_audioreactive_wzrd_quality'

    if 1: # debug: very fast render settings
        H,W          = 1024, 1024
        n_imgs       = 7
        n_steps      = 35
        output_fps   = 12
        seconds_between_keyframes = 7
        inter_frames = int(seconds_between_keyframes * output_fps)

    exp_name = "metropolis_actual"

    for seed in [23,3,41,42]:

        ##############################################################################
        seed_everything(seed)
        img_paths = get_random_img_paths_from_dir(input_dir, n_imgs)
        n = len(img_paths)

        args = StableDiffusionSettings(
            steps = n_steps,
            #ckpt = "juggernaut_XL",
            H = H,
            W = W,
            n_frames = inter_frames*(n-1) + n,
            guidance_scale = 8,
            text_input = "real2real",  # text_input is also the title, but has no effect on interpolations
            interpolation_texts = None,
            interpolation_seeds = [random.randint(1, 1e8) for _ in range(n)],
            interpolation_init_images = img_paths,
            interpolation_init_images_min_strength = random.choice([0.05]),
            interpolation_init_images_max_strength = 0.8,
            n_anchor_imgs = 5,
            latent_blending_skip_f = [0.05, 0.65],
            loop = True,
            n_film = 0,
            fps = output_fps,
            seed = seed,
            clip_interrogator_mode = "fast",
        )

        if args.loop:
            args.n_frames = inter_frames*((n+1)-1) + (n+1)

        # Render the frames:
        frames_dir = real2real_x(W, H, args, img_paths, outdir, n,
                    exp_name = exp_name, audio_path = audio_path, 
                    save_phase_data = 1,
                    save_distance_data = 1)

        # Add post processing audio modulation:
        n_film = 1  # set n_film to 0 to disable FILM interpolation
        post_process_audio_reactive_video_frames(frames_dir, audio_path, output_fps, n_film)