import os, time, random, sys, shutil
sys.path.append('..')

from settings import StableDiffusionSettings
from generation import *

def find_lora_dir(lora_foldername, lora_root_dir = "/home/xander/Projects/cog/lora/exps"):
    # recursively crawl all the subdirectories inside lora_root_dir
    # and return the path that matches lora_foldername
    for subdir, dirs, files in os.walk(lora_root_dir):
        if lora_foldername in subdir:
            return subdir




def real2real_lora(input_images, lora_paths, interpolation_texts, outdir, 
    args = None, 
    seed = int(time.time()), 
    name_str = "",
    force_timepoints = None,
    save_video = True,
    remove_frames_dir = 0,
    save_phase_data = False,  # save condition vectors and scale for each frame (used for later upscaling)
    save_distance_data = 1,  # save distance plots to disk
    keyframe_offset = 0,
    debug = False):

    random.seed(seed)
    
    name = f"real2real_{name_str}_{seed}_{int(time.time())}"
    frames_dir = os.path.join(outdir, name)
    os.makedirs(frames_dir, exist_ok=True)

    if keyframe_offset != 0: # shift the keyframe conditionings
        input_images = input_images[keyframe_offset:]
        interpolation_texts = interpolation_texts[keyframe_offset:]
        lora_paths = lora_paths[keyframe_offset:]
    
    n = len(input_images)

    if args is None:
        args = StableDiffusionSettings(
            #watermark_path = "../assets/eden_logo.png",
            text_input = "real2real",  # text_input is also the title, but has no effect on interpolations
            interpolation_seeds = [random.randint(1, 1e8) for _ in range(n)],
            interpolation_texts = interpolation_texts,
            interpolation_init_images = input_images,
            interpolation_init_images_use_img2txt = True,
            interpolation_init_images_power = 3.0,
            interpolation_init_images_min_strength = 0.25,  # a higher value will make the video smoother, but allows less visual change / journey
            interpolation_init_images_max_strength = 0.97,
            latent_blending_skip_f = [0.05, 0.65],
            guidance_scale = 10,
            n_frames = 52*(n-1) + n,
            #n_frames = 7*(n-1) + n,
            loop = True,
            smooth = True,
            n_film = 0,
            fps = 9,
            steps = 70,
            sampler = "euler",
            seed = seed,
            H = 960,
            W = 960,
            upscale_f = 1.0,
            clip_interrogator_mode = "full",
            lora_scale = 0.8,
            lora_paths = lora_paths,
            lora_path = lora_paths[0],
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

    # run the interpolation and save each frame
    frame_counter = 0
    for frame, t_raw in make_interpolation(args, force_timepoints=force_timepoints):
        frame.save(os.path.join(frames_dir, "frame_%018.14f_%05d.jpg"%(t_raw + keyframe_offset, frame_counter)), quality=95)
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

    # Save a .json with the input_image filenames:
    with open(os.path.join(frames_dir, "input_images.json"), "w") as f:
        json.dump(input_images, f)

    if remove_frames_dir:
        shutil.rmtree(os.path.dirname(frames_dir))
        frames_dir = None

    return video_filename, frames_dir

def replace_nones_with_closest(lora_paths):
    # for all lora_paths that are None, replace them with lora_path left of them:
    lora_paths_shifted_left = [None] + lora_paths[:-1]
    lora_paths = [lora_path if lora_path is not None else lora_path_left for lora_path, lora_path_left in zip(lora_paths, lora_paths_shifted_left)]

    # for all lora_paths that are None, replace them with lora_path right of them:
    lora_paths_shifted_right = lora_paths[1:] + [None]
    lora_paths = [lora_path if lora_path is not None else lora_path_right for lora_path, lora_path_right in zip(lora_paths, lora_paths_shifted_right)]

    return lora_paths

def get_txts_and_loras(json_paths, always_use_lora = True):
    interpolation_texts = []
    lora_paths = []
    ckpts = []

    for json_path in json_paths:
        if json_path is None:
            lora_paths.append(None)
            interpolation_texts.append(None)
            ckpts.append('dreamlike-art/dreamlike-photoreal-2.0')
        else:
            print(json_path)
            with open(json_path) as f:
                data = json.load(f)
                interpolation_texts.append(data['text_input'])
                ckpts.append(data['ckpt'])
                lora_path_orig = data['lora_path']
                lora_path = find_lora_dir(os.path.basename(os.path.dirname(lora_path_orig)))

                if lora_path is not None:
                    lora_paths.append(lora_path + "/final_lora.safetensors")
                else:
                    print("Could not find LORA for: ", json_path)
                    lora_paths.append(None)

    if always_use_lora:
        lora_paths = replace_nones_with_closest(lora_paths)

    return interpolation_texts, lora_paths, ckpts, False
    

if __name__ == "__main__":

    outdir = "results_karo"
    lora_interpolation_dir = '/home/xander/Projects/cog/eden-sd-pipelines/eden/xander/images/karo/gd/interp'
    keyframe_offset = 0

    # Load all .jsons and .jpgs in the lora_interpolation_dir
    jpgs  = sorted([os.path.join(lora_interpolation_dir, f) for f in os.listdir(lora_interpolation_dir) if f.endswith('.jpg')])
    jsons = []
    for jpg in jpgs:
        json_path = jpg.replace('.jpg', '.json')
        if os.path.exists(json_path):
            jsons.append(json_path)
        else:
            print(f"No .json found for {jpg}")
            jsons.append(None)
    jsons = sorted([os.path.join(lora_interpolation_dir, f) for f in os.listdir(lora_interpolation_dir) if f.endswith('.json')])

    print(len(jsons), len(jpgs))
    assert len(jsons) == len(jpgs)

    n = len(jsons)
    n = 2

    for i in [2,4]:

        seed = i
        random.seed(seed)
        indices = random.sample(range(len(jsons)), n)
        #indices = list(range(n))
        input_images = [jpgs[i] for i in indices]
        input_jsons = [jsons[i] for i in indices]

        if 1: # print full debug stacktrace
            interpolation_texts, lora_paths, ckpts, abort = get_txts_and_loras(input_jsons)
            assert len(interpolation_texts) == len(lora_paths) == len(ckpts) == len(input_images)

            ant_img = "/home/xander/Projects/cog/eden-sd-pipelines/eden/xander/images/karo/gd/ant.jpg"
            input_images = [ant_img] + input_images
            lora_paths   = lora_paths + lora_paths
            interpolation_texts = [None] + interpolation_texts

            print(lora_paths)

            real2real_lora(input_images, lora_paths, interpolation_texts, outdir, seed = seed, keyframe_offset=keyframe_offset)
        else:
            try:
                interpolation_texts, lora_paths, ckpts, abort = get_txts_and_loras(input_jsons)
                assert len(interpolation_texts) == len(lora_paths) == len(ckpts) == len(input_images)
                real2real_lora(input_images, lora_paths, interpolation_texts, outdir, seed = seed, keyframe_offset=keyframe_offset)
            except Exception as e:
                #print(input_jsons)
                print(e)
                continue
