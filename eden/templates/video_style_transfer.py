import os, time, random, sys, shutil
sys.path.append('..')

from settings import StableDiffusionSettings
from generation import *


def transfer_video_style(interpolation_texts, input_video_frames_dir, outdir,
    args = None, 
    seed = int(time.time()), 
    name_str = "",
    force_timepoints = None,
    save_video = True,
    remove_frames_dir = 0,
    save_phase_data = False,  # save condition vectors and scale for each frame (used for later upscaling)
    save_distance_data = 1,  # save distance plots to disk
    debug = False):

    random.seed(seed)
    n = len(interpolation_texts)
    
    name = f"real2real_{name_str}_{seed}_{int(time.time()*100)}"
    frames_dir = os.path.join(outdir, name)
    os.makedirs(frames_dir, exist_ok=True)

    video_seed_frames = sorted([os.path.join(input_video_frames_dir, f) for f  in os.listdir(input_video_frames_dir) if '.jpg' in f])

    if args is None:
        args = StableDiffusionSettings(
            text_input = "real2real",  # text_input is also the title, but has no effect on interpolations
            interpolation_seeds = [random.randint(1, 1e8) for _ in range(len(interpolation_texts))],
            interpolation_texts = interpolation_texts,
            interpolation_init_images = video_seed_frames,
            interpolation_init_images_power = 1.0,
            interpolation_init_images_min_strength = 0.6,  # a higher value will make the video smoother, but allows less visual change / journey
            interpolation_init_images_max_strength = 0.6,
            init_image_strength = 0.60,
            latent_blending_skip_f = [0.4, 0.4],
            guidance_scale = 8,
            n_frames = 24*(n-1) + n,
            n_anchor_imgs = 12,
            loop = True,
            smooth = True,
            n_film = 0,
            fps = 9,
            steps = 100,
            seed = seed,
            H = 768,
            W = 960,
            lora_path = None,
        )

    # always make sure these args are properly set:
    args.frames_dir = frames_dir
    args.save_distance_data = save_distance_data
    args.save_phase_data = save_phase_data

    if debug: # overwrite some args to make things go FAST
        args.W, args.H = 512, 512
        args.steps = 25
        args.n_frames = 8*n

    # Only needed when visualising the smoothing algorithm (debugging mode)
    if args.save_distances_to_dir:
        args.save_distances_to_dir = os.path.join(frames_dir, args.save_distances_to_dir)
        os.makedirs(args.save_distances_to_dir, exist_ok=True)
    
    start_time = time.time()
    timepoints = []

    # run the interpolation and save each frame
    frame_counter = 0
    for frame, t_raw in video_style_transfer(args, force_timepoints=force_timepoints):
        frame.save(os.path.join(frames_dir, "frame_%018.14f_%05d.jpg"%(t_raw, frame_counter)), quality=95)
        timepoints.append(t_raw)
        frame_counter += 1

    # run FILM postprocessing (frame blending)
    if args.n_film > 0:
        from film import interpolate_FILM
        frames_dir = interpolate_FILM(frames_dir, args.n_film)
        args.fps = args.fps * (args.n_film + 1)

    if save_video:
        # save video
        loop = (args.loop and len(args.interpolation_seeds) == 2)
        video_filename = f'{outdir}/{name}.mp4'
        write_video(frames_dir, video_filename, loop=loop, fps=args.fps)
    else:
        video_filename = None

    # save settings
    settings_filename = f'{outdir}/{name}.json'
    args.total_render_time = "%.1f seconds" %(time.time() - start_time)
    args.avg_render_time_per_frame = "%.1f seconds" %((time.time() - start_time) / frame_counter)
    save_settings(args, settings_filename)

    if remove_frames_dir:
        shutil.rmtree(os.path.dirname(frames_dir))
        frames_dir = None

    return video_filename, frames_dir, timepoints

    

if __name__ == "__main__":

    outdir = "results"
    input_video_frames_dir = "/home/xander/Projects/cog/stable-diffusion-dev/eden/xander/img2img_inits/video/urbex/frames"
    interpolation_texts = get_prompts_from_json_dir('/home/xander/Projects/cog/eden-sd-pipelines/eden/xander/images/COOPER_final_imgs/seq')
    print(len(interpolation_texts))

    seed = int(time.time())
    seed = 2

    transfer_video_style(interpolation_texts, input_video_frames_dir, outdir, seed = seed)