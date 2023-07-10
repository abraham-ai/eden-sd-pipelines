import os, random, sys
sys.path.append('..')
from settings import StableDiffusionSettings
from generation import *
from templates.interpolate_real2real import real2real
from planner import Planner


def real2real_x(W, H, args, input_images, outdir, n, exp_name = "", audio_feature_path = None, audio_path = None, save_phase_data = False, save_distance_data = False):

    name_str = f"seed_{args.seed}_{exp_name}"

    if audio_feature_path is not None:
        n_final_frames = args.n_frames
        args.planner = Planner(audio_feature_path, args.fps, n_final_frames)

    video_path, frames_dir, timepoints = real2real(input_images, outdir, 
                name_str = name_str, args = args, seed = args.seed, remove_frames_dir = False, 
                save_video = 1, save_phase_data=save_phase_data, save_distance_data = save_distance_data)
    
    if audio_path is not None:
        print("adding audio...")
        add_audio_to_video(audio_path, video_path, video_path.replace(".mp4", "_audio.mp4"))

    return frames_dir

def get_random_img_paths_from_dir(directory_path, n_imgs, seed = 0):
    random.seed(seed)
    img_exts = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]
    all_img_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
    all_img_paths = [f for f in all_img_paths if os.path.splitext(f)[1] in img_exts]
    return random.sample(all_img_paths, n_imgs)

from audioreactive_post_process_frames import post_process_audio_reactive_video_frames


def render_real2real_audioreactive(
    img_paths, # ordered list of filepaths to the keyframe imgs
    audio_feature_path,

    W = 768,
    H = 768,
    audio_path = None,
    out_video_path = None, # where to put the final output video
    seed = 0, 
    inter_frames = 120,  # number of frames to interpolate between each pair of keyframe images
    n_diffusion_steps_per_frame = 32,
    output_fps   = 12,   # final fps will be twice this when n_film = 1
    n_film = 0,         # n_film = 1: apply FILM smoothing (double framerate)
    outdir = 'results/real2real_audioreactive', # tmp dir to store the rendered frames
    debug = False, # call with debug = True to do a quick, fast render
):

    seed_everything(seed)

    if debug:
        W, H = 640, 640
        n_diffusion_steps_per_frame = 24
        inter_frames = 36

    n = len(img_paths)

    args = StableDiffusionSettings(
        steps = n_diffusion_steps_per_frame,
        ckpt = "eden:eden-v1",
        H = H,
        W = W,
        n_frames = inter_frames*(n-1) + n,
        guidance_scale = 7,
        text_input = "real2real",  # text_input is also the title, but has no effect on interpolations
        interpolation_texts = None,
        uc_text = "watermark, nude, text, ugly, tiling, out of frame, blurry, grainy, signature, cut off, draft",  # negative prompting
        interpolation_seeds = [random.randint(1, 1e8) for _ in range(n)],
        interpolation_init_images = img_paths,
        interpolation_init_images_use_img2txt = True,
        interpolation_init_images_power = random.choice([3.0]),
        interpolation_init_images_min_strength = random.choice([0.275]),
        interpolation_init_images_max_strength = 0.925,
        n_anchor_imgs = 5,
        latent_blending_skip_f = [0.05, 0.75],
        loop = True,
        n_film = 0,
        fps = output_fps,
        seed = seed,
        clip_interrogator_mode = "fast",
    )

    # Render the frames:
    frames_dir = real2real_x(W, H, args, img_paths, outdir, n,
                exp_name = "", audio_path = audio_path, 
                save_phase_data = 1,
                save_distance_data = 1)

    # Add post processing audio modulation:
    post_process_audio_reactive_video_frames(frames_dir, output_fps, n_film, audio_feature_path, audio_path = audio_path, out_video_path = out_video_path)



if __name__ == "__main__":

    # audio_path is either a path of a .zip file, or a tuple of (audio_features_pickle, audio_mp3_path)
    audio_feature_path = "/home/xander/Projects/cog/xander_eden_stuff/wzrd/tmp_unzip/features.pkl"
    audio_path         = "/home/xander/Projects/cog/xander_eden_stuff/wzrd/tmp_unzip/music.mp3"
    audio_path         = None
    
    # ordered list of keyframe imgs:
    img_paths =  ["../assets/cow.png", "../assets/eden_logo.png"]

    render_real2real_audioreactive(
        img_paths, # ordered list of filepaths to the keyframe imgs
        out_video_path = "render.mp4", # where to put the final output video
        audio_feature_path = audio_feature_path,
        audio_path = audio_path,
        seed = 0, 
        W = 768,
        H = 768,
        debug = 1
    )