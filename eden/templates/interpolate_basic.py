import os, random, sys, time
sys.path.append('..')

from settings import StableDiffusionSettings
from generation import *
from eden_utils import *
from prompts import text_inputs

def lerp(interpolation_texts, outdir, 
    args = None, 
    seed = int(time.time()), 
    interpolation_seeds = None,
    name_str = "",
    save_phase_data = False,     # save condition vectors and scale for each frame (used for later upscaling)
    save_distance_data = False,  # save distance plots to disk
    debug = False):

    seed_everything(seed)
    n = len(interpolation_texts)
    
    name = f"prompt2prompt_{int(time.time()*100)}_seed_{seed}_{name_str}"
    frames_dir = os.path.join(outdir, name)
    os.makedirs(frames_dir, exist_ok=True)
    
    args = StableDiffusionSettings(
        text_input = interpolation_texts[0],
        interpolation_texts = interpolation_texts,
        interpolation_seeds = interpolation_seeds if interpolation_seeds else [random.randint(1, 1e8) for i in range(n)],
        n_frames = 24*n,
        scale = 10,
        scale_modulation = 0.0,
        loop = True,
        smooth = True,
        latent_blending_skip_f = [0.0, 0.70],
        n_film = 0,
        fps = 9,
        steps = 40,
        sampler = "euler",
        seed = seed,
        W = 768,
        H = 512,
    )

    # always make sure these args are properly set:
    args.frames_dir = frames_dir
    args.save_distance_data = save_distance_data
    args.save_phase_data = save_phase_data

    if debug: # overwrite some args to make things go FAST
        args.W, args.H = 512, 512
        args.steps = 25
        args.n_frames = 8*n

    # run the interpolation and save each frame
    for frame, t_raw in make_interpolation(args):
        frame.save(os.path.join(frames_dir, "frame_%0.16f.jpg"%t_raw), quality=95)

    # run FILM
    if args.n_film > 0:
        from film import interpolate_FILM
        interpolate_FILM(frames_dir, args.n_film)
        frames_dir = os.path.join(frames_dir, "interpolated_frames")

    # save video
    loop = (args.loop and len(args.interpolation_seeds) == 2)
    video_filename = f'{outdir}/{name}.mp4'
    write_video(frames_dir, video_filename, loop=loop, fps=args.fps)

    # save settings
    settings_filename = f'{outdir}/{name}.json'
    save_settings(args, settings_filename)


if __name__ == "__main__":

    outdir = "results"
    n = 4
    seed = int(time.time())

    seed_everything(seed)
    interpolation_texts = random.sample(text_inputs, n)
    lerp(interpolation_texts, outdir, seed=seed, save_distance_data=True, interpolation_seeds=None)
