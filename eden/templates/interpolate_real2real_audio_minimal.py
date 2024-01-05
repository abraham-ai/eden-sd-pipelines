import os, random, sys
sys.path.append('..')
from settings import StableDiffusionSettings
from generation import *
from templates.interpolate_real2real import real2real
from planner import Planner
from audio_post import post_process_audio_reactive_video_frames

def get_random_img_paths_from_dir(directory_path, n_imgs, sorted = False):
    img_exts = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]
    all_img_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
    all_img_paths = [f for f in all_img_paths if os.path.splitext(f)[1] in img_exts]

    paths = random.sample(all_img_paths, n_imgs)

    if sorted: 
        paths.sort()

    return paths


def real2real_x(W, H, args, input_images, outdir, n, 
        exp_name = "", 
        audio_path = None, 
        do_post_process = True,
        update_audio_reactivity_settings = {
                'depth_rescale'     : [105., 255.],
                '3d_motion_xyz'     : [0.7, 0.7, -90],
                'circular_motion_period_s': 15,  # the period of the circular xy motion around the center (in seconds)
                '3d_rotation_xyz'   : [0,0,0],
                'brightness_factor' : 0.003,
                'contrast_factor'   : 0.4,
                'saturation_factor' : 0.5,
                '2d_zoom_factor'    : 0.00,
                'noise_factor'      : 0.0,
    },
        save_phase_data = False, 
        save_distance_data = False):

    name_str = f"seed_{args.seed}_{exp_name}"

    if audio_path is not None:
        n_final_frames = args.n_frames
        args.planner = Planner(audio_path, args.fps, n_final_frames)

    video_path, frames_dir, timepoints = real2real(input_images, outdir, 
                name_str = name_str, args = args, seed = args.seed, remove_frames_dir = False, 
                save_video = 1, save_phase_data=save_phase_data, save_distance_data = save_distance_data)
    
    if do_post_process:
        n_film = 1
        fin_video_path = post_process_audio_reactive_video_frames(frames_dir, audio_path, output_fps, n_film, audio_reactivity_settings)
    else:
        if audio_path is not None:
            print("adding audio...")
            fin_video_path = video_path.replace(".mp4", "_audio.mp4")
            add_audio_to_video(args.planner.audio_path, video_path, fin_video_path)
        else:
            fin_video_path = video_path

    return fin_video_path

if __name__ == "__main__":

    ##############################################################################
    
    # main render settings (eg specified by the user)
    H,W          = 1024+256, 1024+256
    n_steps      = 50       # n_diffusion steps per frame  
    seconds_between_keyframes = 9
    n_imgs       = 30

    # audio_path is either a path of a .zip file, or a tuple of (audio_features_pickle, audio_mp3_path)
    audio_path = ("path_to_features.pkl", "path_to_audio.mp3")
    audio_path = ("/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/audio/versilov/versilov_audio_features_80_40.pkl", "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/audio/versilov/versilov.mp3")
    
    # Get random images from a directory: (this should be replaced with timeline imgs in WZRD)
    input_dir = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/init_imgs/materials"
    input_dir = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/init_imgs/club_long"

    outdir    = 'results_real2real_audioreactive_club_long'

    if 0: # debug: fast render settings
        H,W          = 512, 512
        n_imgs       = 2
        n_steps      = 20
        seconds_between_keyframes = 0.5

    for seed in [0,1]:

        ##############################################################################
        seed_everything(seed)
        if seed % 2 == 0:
            img_paths = get_random_img_paths_from_dir(input_dir, n_imgs, sorted=True)
        else:
            img_paths = get_random_img_paths_from_dir(input_dir, n_imgs, sorted=False)

        n = len(img_paths)
        
        fps = 14
        inter_frames = int(seconds_between_keyframes * fps)

        args = StableDiffusionSettings(
            H = H,
            W = W,
            steps = n_steps,
            n_frames = inter_frames*(n-1) + n,
            guidance_scale = 8,
            text_input = "real2real",  # text_input is also the title, but has no effect on interpolations
            interpolation_seeds = [random.randint(1, 1e8) for _ in range(n)],
            interpolation_init_images = img_paths,
            interpolation_init_images_min_strength = 0.05,
            interpolation_init_images_max_strength = 0.80,
            ip_image_strength = 1.0,
            n_anchor_imgs = 4,
            latent_blending_skip_f = [0.0, 0.8],
            loop = True,
            fps = fps,
            seed = seed,
        )

        args.lpips_max_d = 0.75
        args.lpips_min_d = 0.05

        if args.loop:
            args.n_frames = inter_frames*((n+1)-1) + (n+1)

        real2real_x(W, H, args, img_paths, outdir, n,
                    audio_path = audio_path, 
                    do_post_process = False,
                    save_phase_data = 1,
                    save_distance_data = 1)

    #########################################################################################