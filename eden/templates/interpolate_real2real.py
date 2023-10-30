import os, time, random, sys, shutil, subprocess
sys.path.append('..')

from settings import StableDiffusionSettings
from generation import *
from pipe import pipe_manager

def real2real(
    input_images, 
    outdir, 
    input_texts = None,
    args = None, 
    seed = int(time.time()), 
    name_str = "",
    force_timepoints = None,
    save_video = True,
    remove_frames_dir = False,
    save_phase_data = False,  # save condition vectors and scale for each frame (used for later upscaling)
    save_distance_data = 1,  # save distance plots to disk
    debug = 0):

    random.seed(seed)
    n = len(input_images)
    
    name = f"real2real_{name_str}_{int(time.time()*100)}_{seed}"
    frames_dir = os.path.join(outdir, name)
    os.makedirs(frames_dir, exist_ok=True)
    
    if args is None:
        args = StableDiffusionSettings(
            #watermark_path = "../assets/eden_logo.png",
            text_input = "real2real",  # text_input is also the title, but has no effect on interpolations
            interpolation_seeds = [random.randint(1, 1e8) for _ in range(n)],
            interpolation_texts = input_texts,
            interpolation_init_images = input_images,
            interpolation_init_images_min_strength = 0.05,  # a higher value will make the video smoother, but allows less visual change / journey
            interpolation_init_images_max_strength = 0.90,
            latent_blending_skip_f = random.choice([[0.05, 0.65]]),
            compile_unet = False,
            guidance_scale = random.choice([6]),
            n_anchor_imgs = random.choice([3]),
            sampler = "euler",
            n_frames = 16*n,
            loop = True,
            smooth = True,
            n_film = 1,
            fps = 12,
            steps = 30,
            seed = seed,
            H = random.choice([960]),
            W = random.choice([640]),
            ip_image_strength = random.choice([0.65]),
        )

    # always make sure these args are properly set:
    args.frames_dir = frames_dir
    args.save_distance_data = save_distance_data
    args.save_phase_data = save_phase_data

    if debug: # overwrite some args to make things go FAST
        args.W, args.H = 512, 512
        args.steps = 20
        args.n_frames = 6*n
        args.n_anchor_imgs = 2

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

        print("Clearing SD pipe memory to run FILM...")
        pipe_manager.clear()

        frames_dir = os.path.abspath(frames_dir)
        command = [sys.executable, os.path.join(str(SD_PATH), "eden/film.py"), "--frames_dir", frames_dir, "--times_to_interpolate", str(args.n_film)]

        print("running command:", ' '.join(command))
        result = subprocess.run(command, text=True, capture_output=True)
        print(result)
        print(result.stdout)
        frames_dir = os.path.join(frames_dir, "interpolated_frames")

        args.fps = args.fps * (args.n_film + 1)

    if save_video:
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

    outdir = "results_real2real"

    init_imgs = [
        "https://minio.aws.abraham.fun/creations-stg/7f5971f24bc5c122aed6c1298484785b4d8c90bce41cc6bfc97ad29cc179c53f.jpg",
        "https://minio.aws.abraham.fun/creations-stg/445eebc944a2d44bb5e0337ed4198ebf54217c7c17729b245663cf5c4fea182c.jpg",
        "https://minio.aws.abraham.fun/creations-stg/049848c63707293cddc766b2cbd230d9cde71f5075e48e9e02c6da03566ddae7.jpg",
        ]

    #img_dir = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/init_imgs/01_great_inits"
    #init_imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]

    init_imgs = random.sample(init_imgs, 3)
    real2real(init_imgs, outdir, seed = 0)
