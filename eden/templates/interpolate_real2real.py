import os, time, random, sys, shutil
sys.path.append('..')

from settings import StableDiffusionSettings
from generation import *


def real2real(input_images, outdir, 
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
    n = len(input_images)
    
    name = f"real2real_{name_str}_{int(time.time()*100)}"
    frames_dir = os.path.join(outdir, name)
    os.makedirs(frames_dir, exist_ok=True)

    if args is None:
        args = StableDiffusionSettings(
            #watermark_path = "../assets/eden_logo.png",
            text_input = "real2real",  # text_input is also the title, but has no effect on interpolations
            interpolation_seeds = [random.randint(1, 1e8) for _ in range(n)],
            #interpolation_texts = None,
            interpolation_init_images = input_images,
            interpolation_init_images_use_img2txt = True,
            interpolation_init_images_power = 3.0,
            interpolation_init_images_min_strength = 0.3,  # a higher value will make the video smoother, but allows less visual change / journey
            interpolation_init_images_max_strength = 0.95,
            latent_blending_skip_f = [0.15, 0.75],
            guidance_scale = 8,
            scale_modulation = 0.0,
            n_frames = 24*n,
            loop = True,
            smooth = True,
            n_film = 0,
            fps = 9,
            steps = 40,
            sampler = "euler",
            seed = seed,
            H = 576,
            W = 576,
            upscale_f = 1.0,
            clip_interrogator_mode = "fast",
            aesthetic_target             = None,  # None means we'll use the init_images as target
            aesthetic_steps              = 10,
            aesthetic_lr                 = 0.0001,
            ag_L2_normalization_constant = 0.1, # for real2real, only normalize the aesthetic gradient a tiny bit
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
    for frame, t_raw in make_interpolation(args, force_timepoints=force_timepoints):
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
    n = 3

    init_imgs = [
        "https://generations.krea.ai/images/3cd0b8a8-34e5-4647-9217-1dc03a886b6a.webp",
        "https://generations.krea.ai/images/928271c8-5a8e-4861-bd57-d1398e8d9e7a.webp",
        "https://generations.krea.ai/images/865142e2-8963-47fb-bbe9-fbe260271e00.webp"
    ]
    
    seed = int(time.time())
    seed = 2

    random.seed(seed)
    input_images = random.sample(init_imgs, n)

    real2real(input_images, outdir, seed = seed)