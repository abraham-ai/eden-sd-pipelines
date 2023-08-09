import os, random, sys, time, random
sys.path.append('..')

from settings import StableDiffusionSettings
from generation import *
from eden_utils import *
from prompts import text_inputs

def lerp(
    interpolation_texts, 
    outdir, 
    args = None, 
    seed = int(time.time()), 
    interpolation_seeds = None,
    name_str = "",
    save_phase_data = False,     # save condition vectors and scale for each frame (used for later upscaling)
    save_distance_data = False,  # save distance plots to disk
    debug = 0):

    seed_everything(seed)
    n = len(interpolation_texts)
    
    name = f"prompt2prompt_{int(time.time())}_seed_{seed}_{name_str}"
    frames_dir = os.path.join(outdir, name)
    os.makedirs(frames_dir, exist_ok=True)
    
    args = StableDiffusionSettings(
        text_input = interpolation_texts[0],
        interpolation_texts = interpolation_texts,
        interpolation_seeds = interpolation_seeds if interpolation_seeds else [random.randint(1, 1e8) for i in range(n)],
        n_frames = 100*n,
        guidance_scale = random.choice([8]),
        loop = True,
        smooth = True,
        latent_blending_skip_f = random.choice([[0.0, 0.75]]),
        n_anchor_imgs = random.choice([6]),
        n_film = 0,
        fps = 12,
        steps = 60,
        seed = seed,
        H = 960-128,
        W = 1024+1024+256,
    )

    # always make sure these args are properly set:
    args.frames_dir = frames_dir
    args.save_distance_data = save_distance_data
    args.save_phase_data = save_phase_data

    if debug: # overwrite some args to make things go FAST
        args.W, args.H = 640, 640
        args.steps = 20
        args.n_frames = 8*n

    start_time = time.time()

    # run the interpolation and save each frame
    frame_counter = 0
    for frame, t_raw in make_interpolation(args):
        frame.save(os.path.join(frames_dir, "frame_%018.14f_%05d.jpg"%(t_raw, frame_counter)), quality=95)
        frame_counter += 1
        
    # run FILM
    if args.n_film > 0:
        # clear cuda cache:
        torch.cuda.empty_cache()

        print('predict.py: running FILM...')
        FILM_MODEL_PATH = os.path.join(SD_PATH, 'models/film/film_net/Style/saved_model')
        film_script_path = os.path.join(SD_PATH, 'eden/film.py')
        abs_out_dir_path = os.path.abspath(str(frames_dir))
        command = ["python", film_script_path, "--frames_dir", abs_out_dir_path, "--times_to_interpolate", str(args.n_film), '--update_film_model_path', FILM_MODEL_PATH]
        
        run_and_kill_cmd(command)
        print("predict.py: FILM done.")
        frames_dir = Path(os.path.join(abs_out_dir_path, "interpolated_frames"))

    # save video
    loop = (args.loop and len(args.interpolation_seeds) == 2)
    video_filename = f'{outdir}/{name}.mp4'
    write_video(frames_dir, video_filename, loop=loop, fps=args.fps)

    # save settings
    settings_filename = f'{outdir}/{name}.json'
    args.total_render_time = "%.1f seconds" %(time.time() - start_time)
    save_settings(args, settings_filename)


if __name__ == "__main__":

    outdir = "jeffrey_lerp_test_02"
    n = 3

    for i in range(2):
        seed = np.random.randint(0, 1000)
        seed = i

        seed_everything(seed)
        interpolation_texts = random.sample(text_inputs, n)

        text_inputs1 = [

            "a microscopic photo of cosmic water droplets floating in thin air, photorealistic, microscope, flat, blurred background",
            "photo of dewdrops on spiderwebs, leaves, and petals, sparkling like jewels as they begin to trickle downwards in the morning light, photorealistic",
            "a photo of droplets Forming on the grass: A serene landscape at dawn where dew is gently forming on the tips of fresh, green grass, first light of the day. Individual droplets of water are clinging to the grass, reflecting the early morning light, photorealistic",
            "a photo of fresh rainfall: A dense forest enveloped in a sudden downpour. Heavy raindrops hitting large, broad leaves and the surface of puddles, photorealistic",
            "a photo of small forest critters taking shelter from a massive downpour, covering underneath large leafs, National Geographic",
            "a photo of a completely soaked forest, drenched in water during a massive downpour, the water is forming huge puddles and trenches and is starting to flow, photorealistc",
            "a dslr photo of a flowing stream, small bubbling mountain stream, meandering through a rocky landscape high in the mountains. Clear water rushes over smooth stones, banks covered in wildflowers and ferns. The sunlight is filtering through the leaves, photorealistic.",
            "a photo of the river's mighty journey: A young river winding through a majestic valley with huge mountains, cutting through cliffs and forests. It's seen from a distance. National Geographic wallpaper, photorealistic",
            "a photo of a wide river flowing through magnificent landscapes, wide meandering river, sandy banks, wallpaper, photorealistic",
            "a photo of a wide river meeting the Ocean, waves gently lapping at the estuary. The horizon stretches infinitely, and the colors of sunset blend with the blue of the sea. A sailboat is seen in the distance, and seabirds are silhouetted against the sky.",
            "a photo of the mighty blue sea, expanding infinitely in all directions. The sky is cloudy and there are waves with white crests, turbulent skies, thunder, storm, photorealistic",
            "a dark photo of the Ocean Depths: shrouded underwater scene revealing the mysterious, dark depths of the ocean filled with alien creatures, eerie lighting, photorealistic",
            "a photo of a rich coral reef, marine life, fish, and turtle captured in their natural habitat. Shafts of sunlight penetrate the surface, symbolizing the complexity and wonder of life below the surface, photorealistic",
            "a wide angle photo taken exactly at the surface of the ocean water, showing both the air above the water and the coral reef below, photorealistic, award winning, wallpaper",
            "a photo of water evaporating into the sky: A stunning view of a coastal area at sea, where the hot sun shining from above causes water to evaporate from the ocean's surface. The sun casts a golden glow on the water, creating a scene that signifies transition, transformation, and the eternal cycle of water.",
            "a closeup photo of tiny water droplets in the sky and transparent, white steamclouds forming above the surface of the ocean, photorealistic",
            "a photo taken from high above the clouds at the edges of outer space, looking down at the earth where the ocean meets the land, photorealistic, birds eye view, edge of space",
            "a photorealistic artwork showing microscopic water droplets as small worlds of glass, time crystals, photorealistic",
            "a stunning photo of the earth seen from afar, from the international spacestation, with sunrays striking through the atmosphere, photorealistic",
            "a telescope photo of the deep universe: a sky filled with stars, galaxies, and nebulae. The Milky Way stretches across the sky. The scene encapsulates a sense of cosmic connection, mystery, and existential wonder.",
            "incredible photorealistic artwork of the Atman, the Soul: A 3D render of the soul or Atman, eternal in time. mystical symbols converge to form a harmonious pattern, a human silhouette, photorealistic",
            "incredible photorealistic depiction of the ultimate void, the end of time, the circle of life, symbolizing that the cycle continues and life is in perpetual flow and reincarnation",
            "a fantastic 3D render of yin and yang, the balance of life, the balance of the universe, against the cosmic voide, photorealistic",
        
        ]

        text_inputs = [

            " with sharp details of cosmic water droplets floating in thin air, microscope, flat, blurred background",
            " of dewdrops on spiderwebs, leaves, and petals, sparkling like jewels as they begin to trickle downwards in the morning light",
            " of droplets Forming on the grass: A serene landscape at dawn where dew is gently forming on the tips of fresh, green grass, first light of the day. Individual droplets of water are clinging to the grass, reflecting the early morning light",
            " of fresh rainfall: A dense forest enveloped in a sudden downpour. Heavy raindrops hitting large, broad leaves and the surface of puddles",
            " of small forest critters taking shelter from a massive downpour, covering underneath large leafs, National Geographic",
            " of a completely soaked forest, drenched in water during a massive downpour, the water is forming huge puddles and trenches and is starting to flow",
            " of a flowing stream, small bubbling mountain stream, meandering through a rocky landscape high in the mountains. Clear water rushes over smooth stones, banks covered in wildflowers and ferns. The sunlight is filtering through the leaves.",
            " of the river's mighty journey: A young river winding through a majestic valley with huge mountains, cutting through cliffs and forests. It's seen from a distance. National Geographic wallpaper",
            " of a wide river flowing through magnificent landscapes, wide meandering river, sandy banks, wallpaper",
            " of a wide river meeting the Ocean, waves gently lapping at the estuary. The horizon stretches infinitely, and the colors of sunset blend with the blue of the sea. A sailboat is seen in the distance, and seabirds are silhouetted against the sky.",
            " of the mighty blue sea, expanding infinitely in all directions. The sky is cloudy and there are waves with white crests, turbulent skies, thunder, storm",
            " the Ocean Depths: shrouded underwater scene revealing the mysterious, dark depths of the ocean filled with alien creatures, eerie lighting",
            " of a rich coral reef, marine life, fish, and turtle captured in their natural habitat. Shafts of sunlight penetrate the surface, symbolizing the complexity and wonder of life below the surface",
            " taken exactly at the surface of the ocean water, showing both the air above the water and the coral reef below, award winning, wallpaper",
            " of water evaporating into the sky: A stunning view of a coastal area at sea, where the hot sun shining from above causes water to evaporate from the ocean's surface. The sun casts a golden glow on the water, creating a scene that signifies transition, transformation, and the eternal cycle of water.",
            " of tiny water droplets in the sky and transparent, white steamclouds forming above the surface of the ocean",
            " high above the clouds at the edges of outer space, looking down at the earth where the ocean meets the land, birds eye view, edge of space",
            " showing microscopic water droplets as small worlds of glass, time crystals",
            " of the earth seen from afar, from the international spacestation, with sunrays striking through the atmosphere",
            " of the deep universe: a sky filled with stars, galaxies, and nebulae. The Milky Way stretches across the sky. The scene encapsulates a sense of cosmic connection, mystery, and existential wonder.",
            " of the Atman, the Soul: A 3D render of the soul or Atman, eternal in time. mystical symbols converge to form a harmonious pattern, a human silhouette",
            " of the ultimate void, the end of time, the circle of life, symbolizing that the cycle continues and life is in perpetual flow and reincarnation",
            " of yin and yang, the balance of life, the balance of the universe, against the cosmic voide",
        
        ]

        prefix = "Acid & Hallucinogenic - artwork by Yoko Honda & James Rosenquist"
        prefixes = [
            "3D render, unreal engine, psychedelic visual",
            "depth of field, fractal patterns",
            "geometric shapes, Stereoscopic 3D, Salvador Dali",
            "unity 3D, psychedelic colors, hyperrealism"
        ]

        prefixes  = [
            "Ray Tracing, Mandelbrot Set, Liquid Crystal Display",
            "Virtual Reality, Quantum Physics, H.R. Giger",
            "Augmented Reality, Sacred Geometry, Yayoi Kusama",
            "Motion Capture, Optical Illusions, Wassily Kandinsky",
            "Holographic Projections, Fractal Flames, Alex Grey",
            "3D Scanning, Cubism, Unreal Physics",
            "Cyberpunk, Neon Retro Futurism",
            ]



        interpolation_texts = [prefix + f for f in text_inputs]

        for txt in interpolation_texts:
            print(txt)
            print("-----------------------")
        
        if 0:
            lerp(interpolation_texts, outdir, seed=seed, save_distance_data=True, interpolation_seeds=None)
        else:
            try:
                lerp(interpolation_texts, outdir, seed=seed, save_distance_data=True, interpolation_seeds=None)
            except KeyboardInterrupt:
                print("Interrupted by user")
                exit()  # or sys.exit()
            except Exception as e:
                print(f"Error: {e}")  # Optionally print the error
                continue