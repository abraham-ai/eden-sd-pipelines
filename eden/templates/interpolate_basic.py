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
        n_frames = 64*n,
        guidance_scale = random.choice([8]),
        loop = True,
        smooth = True,
        latent_blending_skip_f = random.choice([[0.05, 0.6]]),
        n_anchor_imgs = random.choice([5]),
        n_film = 0,
        fps = 12,
        steps = 60,
        seed = seed,
        H = 1024,
        W = 1024+640,
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
    for frame, t_raw in make_interpolation(args):
        frame.save(os.path.join(frames_dir, "frame_%0.16f.jpg"%t_raw), quality=95)

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

    outdir = "results_lerp_big"
    n = 3

    for i in range(4):
        seed = np.random.randint(0, 1000)
        #seed = i

        seed_everything(seed)
        interpolation_texts = random.sample(text_inputs, n)

        interpolation_texts = [
                "Stone Age Tools, A prehistoric landscape at dawn with a humanoid figure kneeling on the ground, crafting sharp-edged flint tools amidst scattered rocks and bones. 4k, Photorealistic",
                "Agricultural Revolution, Fields bathed in golden sunlight, where human figures tend to crops using primitive wooden plows, surrounded by early granaries and mud huts. 4k, Photorealistic",
                "Bronze Age and Metalwork, A bustling village market scene with an artisan melting copper and tin in a fiery furnace, fashioning early bronze weapons and tools, while villagers barter goods around. 4k, Photorealistic",
                "Medieval Engineering, A medieval town with towering cathedrals under construction, showing engineers using wooden cranes, pulleys, and early mechanical clocks in the town square. 4k, Photorealistic",
                "Industrial Revolution, Smoky factories with large chimneys dominate a landscape, as steam trains move on tracks and people operate large looming machines, with the distant hum of the steam engine. 4k, Photorealistic",
                "Birth of Electricity, An early 20th-century home interior lit by a single electric bulb, with a family gathered around, marveling at a phonograph playing music, and a telephone on a wooden table. 4k, Photorealistic",
                "Computer Age, A 1970s office scene with workers using large mainframe computers and punch cards, transitioning to early personal computers with floppy disks and green monochrome screens. 4k, Photorealistic",
                "Digital Revolution, A late 90s living room with a family gathered around a bulky desktop computer, dialing up to access the internet, while mobile phones and digital cameras lie on the table. 4k, Photorealistic",
                "Modern Tech Era, A contemporary, minimalist workspace dominated by sleek laptops, tablets, drones hovering outside a window, and augmented reality glasses charging on a desk. 4k, Photorealistic",
                "Futuristic, A serene, holographic living space with translucent walls, where a person interacts with a hovering AI assistant, crafting tools and objects using a 3D nano-assembler, and the ambiance filled with soft glows of quantum computers. 4k, Photorealistic",
        ]

        interpolation_texts_2 = [
            "A vast empty landscape bathed in the soft glow of dawn, the horizon stretches infinitely with a rich soil bed, and nestled within it is a solitary, firm seed with its outer shell beginning to crack.",
            "In the middle of a sunlit clearing, a tiny sapling emerges from the earth, its first green leaves tenderly reaching out towards the sun's nurturing light.",
            "A young tree stands proudly amidst a meadow, its roots anchored deep and branches starting to spread out, providing shade to the ground below and a perch for a bird that sits atop.",
            "In a bustling forest, the tree has now grown to a significant height, its thick bark bearing the marks of time, while a squirrel scampers up its sturdy trunk.",
            "The tree, fully grown, dominates the scene, its expansive canopy sheltering a diverse array of life, from birds that nest within its branches to ferns growing at its base.",
            "A golden-hued autumn landscape where the tree, its leaves transformed into shades of red and gold, drops a colorful blanket onto the ground, heralding the change of seasons.",
            "Winter's embrace covers the scene. The tree, now barren of leaves, stands resilient against a backdrop of snow, its silhouette contrasting against the pale sky, as snowflakes settle on its branches.",
            "Time has passed, and the majestic tree now stands with a hollow in its trunk, giving refuge to a family of owls, while mushrooms and mosses claim spaces on its aging bark.",
            "An evening scene portrays the tree in twilight years, branches withered and fewer leaves, casting long shadows under the setting sun, with fallen logs surrounding it, a testament to time's passage.",
            "The once mighty tree, now a mere stump, provides a platform for new life: a cluster of saplings sprouting from its center, reaching for the sky, heralding a new beginning.",
        ]

        for txt in interpolation_texts:
            print(txt)
            print("-----------------------")
        
        if 1:
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