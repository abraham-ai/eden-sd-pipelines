import os, random, sys, tempfile
import datetime, time
sys.path.append('..')
from settings import StableDiffusionSettings
from generation import *
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


def real2real_audioreactive(
        input_images, audio_path, 
        render_settings = {
            "W": 640,
            "H": 640,
            "steps": 25,
            "seconds_between_keyframes": 12,
            "fps": 10,
        },
        audio_reactivity_settings = {
                'depth_rescale'     : [105., 255.],    # rescale depth map to this range # dont change!!
                '3d_motion_xyz'     : [0.7, 0.7, -90], # camera motion limits
                'circular_motion_period_s': 15,  # the period of the circular xy motion around the center (in seconds)
                '3d_rotation_xyz'   : [0,0,0],
                'brightness_factor' : 0.0005, # 0.001
                'contrast_factor'   : 0.3, #0.4
                'saturation_factor' : 0.4, #0.5
                '2d_zoom_factor'    : 0.00,
                'noise_factor'      : 0.0},
        output_dir = None,
        save_distance_data = True, # save perceptual distance plots to disk
        loop = True, # wrap the video to make it loop-able
        seed = 0,
        interpolation_seeds = None,
        debug = False
        ):

    start_time = time.time()
    random.seed(seed)
    n = len(input_images)

    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    os.makedirs(output_dir, exist_ok=True)

    if debug: # debug: fast render settings
        render_settings["H"]     = 512
        render_settings["W"]     = 512
        render_settings['steps'] = 15
        n = 6
        input_images = input_images[:n]
        render_settings["seconds_between_keyframes"] = 15

    # Compute number of frames between keyframes:
    inter_frames = int(render_settings["seconds_between_keyframes"] * render_settings["fps"])

    args = StableDiffusionSettings(
        H = render_settings["H"],
        W = render_settings["W"],
        steps = render_settings['steps'],
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

    args.planner = Planner(audio_path, args.fps, args.n_frames)

    name = f"real2real_audio_{int(time.time()*100)}_{seed}"
    frames_dir = os.path.join(output_dir, name)
    os.makedirs(frames_dir, exist_ok=True)
    
    # always make sure these args are properly set:
    args.frames_dir = frames_dir
    args.save_distance_data = save_distance_data

    # run the real2real interpolation and save each frame to disk:
    frame_counter = 0
    for frame, t_raw in make_interpolation(args):
        frame.save(os.path.join(frames_dir, "frame_%018.14f_%05d.jpg"%(t_raw, frame_counter)), quality=95)
        frame_counter += 1

    # apply audio reactive post processing:
    print("Frames rendered, applying audio reactive post processing...")
    fin_video_path = post_process_audio_reactive_video_frames(frames_dir, args.planner, args.fps, 
        n_film = 1,
        update_audio_reactivity_settings = audio_reactivity_settings)

    # save settings
    settings_filename = f'{output_dir}/{name}.json'
    args.total_render_time = "%.1f seconds" %(time.time() - start_time)
    args.avg_render_time_per_frame = "%.1f seconds" %((time.time() - start_time) / frame_counter)
    save_settings(args, settings_filename)

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
    fin_video_path = real2real_audioreactive(img_paths, audio_path, output_dir = output_dir, seed=seed, debug = True)