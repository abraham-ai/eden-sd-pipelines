import sys
sys.path.append('..')

import json
import os
import random
from PIL import Image
import moviepy.editor as mpy

from settings import *
from generation import *
from prompts import *
from eden_utils import *


checkpoint_options = [
    "runwayml:stable-diffusion-v1-5",
    "dreamlike-art:dreamlike-photoreal-2.0",
    "eden:eden-v1"
]

def generate_basic(
    text_input, 
    outdir, 
    seed = int(time.time()),
    debug = False,
    init_image_data = None,
    prefix = "",
    suffix = ""):

    img_dir = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/memes"
    #img_dir = "/data/xander/Projects/cog/stable-diffusion-dev/eden/xander/init_imgs/test"
    img_paths = [f for f in os.listdir(img_dir) if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg")]
    init_img = os.path.join(img_dir, random.sample(img_paths, 1)[0])

    args = StableDiffusionSettings(
        #ckpt = "stable-diffusion-xl-base-1.0",
        mode = "generate",
        W = random.choice([1024]),
        H = random.choice([1024]),
        sampler = random.choice(["euler", "euler_ancestral"]),
        steps = 35,
        guidance_scale = random.choice([7]),
        upscale_f = random.choice([1.0]),
        text_input = text_input,
        seed = seed,
        n_samples = 1,
        lora_path = None,
        init_image_data = init_img,
        init_image_strength = random.choice([0.7]),
        #control_guidance_start = random.choice([0.0, 0.0, 0.05, 0.1]),
        #control_guidance_end = random.choice([0.5, 0.6, 0.7]),
        #control_guidance_end = random.choice([0.65]),
        #controlnet_path = "controlnet-luminance-sdxl-1.0", 
        #controlnet_path = "controlnet-depth-sdxl-1.0-small",
        #controlnet_path = "controlnet-canny-sdxl-1.0-small",
        #controlnet_path = "controlnet-canny-sdxl-1.0",
        #controlnet_path = random.choice(["controlnet-luminance-sdxl-1.0", "controlnet-luminance-sdxl-1.0", "controlnet-luminance-sdxl-1.0", "controlnet-luminance-sdxl-1.0", "controlnet-canny-sdxl-1.0", "controlnet-depth-sdxl-1.0-small"]),
    )

    controlnet_img = load_img(args.init_image_data, "RGB")

    #name = f'{prefix}{args.text_input[:40]}_{os.path.basename(args.lora_path)}_{args.seed}_{int(time.time())}{suffix}'
    name = f'{prefix}{args.text_input[:40]}_{args.seed}_{int(time.time())}{suffix}'
    name = name.replace("/", "_")
    
    generator = make_images(args)

    os.makedirs(outdir, exist_ok = True)
    save_control_img = True

    for i, img in enumerate(generator):
        frame = f'{name}_{i}.jpg'
        img.save(os.path.join(outdir, frame), quality=95)
        if save_control_img:
            control_frame = f'{name}_{i}_cond.jpg'
            controlnet_img.save(os.path.join(outdir, control_frame), quality=95)

    # save settings
    settings_filename = f'{outdir}/{name}.json'
    save_settings(args, settings_filename)


if __name__ == "__main__":

    
    outdir = "results"
    
    meme_prompts = [
        "Psychedelic dreamscape with neon accents",
        "Vintage comic book textures",
        "Cyberpunk cityscape at night",
        "Graffiti wall meets abstract expressionism",
        "Retro 80s vaporwave aesthetic",
        "Watercolor painting with impressionistic strokes",
        "Pointillism in pastel colors",
        "Victorian tapestry meets modern geometry",
        "Japanese ukiyo-e with digital glitches",
        "Black and white film noir atmosphere",
        "A medieval knight lost in a futuristic city"
        "Alice in Wonderland falling through a Dali painting",
        "Sherlock Holmes solving mysteries in a cyberpunk Tokyo",
        "Mona Lisa joining a masquerade ball",
        "Pirate captain navigating through a sea of digital pixels",
        "Einstein pondering equations amidst a Van Gogh starry night",
        "A jazz musician in an Escher-like world of impossible geometry",
        "A spaceman wandering in an Art Nouveau garden",
        "A vampire lurking in a graffiti-covered alley",
        "A superhero in a chaotic cubist landscape",
        "A ninja riding a unicorn through a galaxy made of pizza slices",
        "A werewolf howling at a disco ball moon",
        "Bob Ross painting happy little trees on Mars",
        "Zombie Shakespeare performing on Broadway",
        "Caveman discovering fire in a world made of ice cream",
        "Gandalf the Grey doing stand-up comedy in an underwater Atlantis",
        "A robot participating in an Ancient Greek Olympics",
        "Alien abducting a cow from a Salvador Dali landscape",
        "Mermaid singing opera in a cybernetic forest",
        "Freddie Mercury duetting with a holographic dragon",
        "An astronaut breakdancing on a Rubik's Cube planet",
        "Vampire enjoying a day at the beach with sunscreen",
        "Elvis Presley leading a UFO invasion",
        "An archaeologist uncovering the Internet as a buried artifact",
        "Queen Cleopatra DJing at a rave in a pyramid",
        "Santa Claus competing in a drag race on his sleigh",
        "Ghost of Steve Jobs presenting the next iPhone in Valhalla",
        "Frankenstein's monster participating in a beauty pageant",
        "Spartan warrior taking part in a breakdance battle",
        "Sigmund Freud analyzing the dreams of a sentient cactus",
        "Time-traveling T-Rex playing electric guitar at Woodstock",
        "Nicolas Cage as every character in a 'Last Supper' made of laser beams",
        "Michelangelo's David doing the moonwalk on a tightrope between two black holes",
        "Emoji-faced Shakespeare rapping in a duel against a beatboxing Plato",
        "Marilyn Monroe as a cybernetic spy in a world where cats rule humans",
        "An army of rubber duckies conquering a miniature Tokyo Godzilla-style",
        "Oscar Wilde and Kurt Cobain in a lightsaber duel on the moon",
        "Albert Einstein and Stephen Hawking in a cosmic chess match where pawns are tiny universes",
        "Beyoncé leading a revolution in a kingdom of sentient emojis",
        "A velociraptor riding a flying narwhal through an asteroid field of donuts",
        "Da Vinci's Vitruvian Man breakdancing in a quantum foam disco",
        "Frida Kahlo and Andy Warhol in a graffiti war on the Berlin Wall",
        "Homer Simpson and Peter Griffin as buddy cops in a dimension made of bacon",
        "Area 51 guards joining forces with Hogwarts wizards to repel an alien invasion",
        "Napoleon Bonaparte participating in a hotdog eating contest against Cthulhu",
        "Charlie Chaplin and Darth Vader in a tap dance showdown on Mars",
        "Keanu Reeves as Neo fighting Agent Smiths made entirely of memes",
        "A sentient avocado proposing world peace at the United Nations",
        "Captain Jack Sparrow and Han Solo in a space race to find the Holy Grail",
        "Mona Lisa and SpongeBob SquarePants as rival contestants on a reality TV show set in Valhalla",
        "A samurai at a karaoke bar, visibly sweating as he tries to hit a high note",
        "A wizard in a dueling circle, his wand backfiring, eyes bulging in disbelief",
        "A soccer player missing a penalty kick, face contorting like a Picasso painting",
        "A spy caught in the act, dripping in neon-colored sweat under a blacklight",
        "A chef in a kitchen inferno, flames reflecting in his tear-filled eyes",
        "A superhero stuck in quicksand, cape tattered, emitting visible 'help me' thought bubbles",
        "A bodybuilder lifting a dumbbell made of rubber ducks, veins pulsating in comic exaggeration",
        "A student during finals week, drowning in a sea of books, tears forming a waterfall",
        "An office worker spilling coffee, time slowing, face twisting in slow-motion horror",
        "A rockstar breaking a guitar string mid-solo, eyes popping out like a Tex Avery cartoon",
        "A hairy caveman discovering fire only to burn his beard, eyes widening in disbelief",
        "A dog making a funny face while riding a skateboard, tongue flailing wildly",
        "A cute kitten acting as a DJ, paws scratching a vinyl record, eyes fixated on the beat",
        "A stoned chiller floating on a cloud of smoke, eyes red as cherries, lost in a bag of chips",
        "A drunk man dancing with a broom as if it's the love of his life, sweat flying in hearts",
        "Crazy eyes girlfriend cutting a 'love you forever' message into a tree, eyes spinning like pinwheels",
        "A robot butler spilling drinks, sparks flying, looking more distressed than its human guests",
        "An overly enthusiastic gym trainer lifting weights that are actually balloons, popping them mid-lift",
        "A hipster vampire sipping on a blood latte, eyes rolling to the back of his head in peak irony",
        "A conspiracy theorist wrapped in tinfoil, eyes bulging at the sight of a fake UFO",
        "An influencer zombie taking a selfie while munching on a brain, #BrainFood",
        "A UFO abducting a cow, only for the cow to abduct the alien right back, role reversal!",
        "A pirate with a peg leg and a hook hand trying to tie his shoes, eyes popping in frustration",
        "A ninja throwing glitter bombs instead of smoke bombs, utterly baffled at his own fabulousness",
        "A grandmother in a jetpack, knitting at supersonic speed, yarn trailing like contrails",
        "A hipster mermaid vaping seaweed, eyes so rolled they're looking at her own brain",
        "A super chill Buddha breakdancing on a spinning lotus, with disco lights for auras",
        "An astronaut planting a fast-food flag on an alien planet, realizing it's made entirely of broccoli",
        "A Sherlock Holmes with magnifying glasses for eyes, still unable to find where he left his keys",
        "A talking donut preaching fitness, sweating glaze, while lifting mini-dumbbells",
        "everyone looks like Nicolas Cage, Nicolas Cage faces everywhere",
        "everyone looks like Nicolas Cage, Nicolas Cage faces everywhere",
        "everyone looks like Nicolas Cage, Nicolas Cage faces everywhere",
        "everyone looks like Captain Jack Sparrow, Captain Jack Sparrow faces everywhere",
        "everyone looks like Captain Jack Sparrow, Captain Jack Sparrow faces everywhere",
        "everyone looks like Captain Jack Sparrow, Captain Jack Sparrow faces everywhere",
    ]

    for i in range(2000):
        seed = int(time.time())
        #seed = i
        
        seed_everything(seed)

        p2 = list(set(get_prompts_from_json_dir("/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/good_controlnet_jsons")))
        all_p = list(set(text_inputs + sdxl_prompts + p2 + meme_prompts))
        text_input = random.choice(all_p)

        if random.random() < 0.1:
            text_input = "remix_this_image"

        print(text_input)
        if 0:
            generate_basic(text_input, outdir, seed = seed)
        else:
            try:
                generate_basic(text_input, outdir, seed = seed)
            except KeyboardInterrupt:
                print("Interrupted by user")
                exit()  # or sys.exit()
            except Exception as e:
                print(f"Error: {e}")  # Optionally print the error
                continue