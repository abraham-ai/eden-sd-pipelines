import sys
sys.path.append('..')

import json
import os
import random
from PIL import Image
import moviepy.editor as mpy

from settings import StableDiffusionSettings
from generation import *
from prompts import text_inputs, style_modifiers
from eden_utils import *


modifiers = [
    "trending on artstation",
    "fish eye lens",
    "polaroid photo",
    "poster by ilya kuvshinov katsuhiro", 
    "by magali villeneuve",
    "by jeremy lipkin",
    'by jeremy mann',
    'by jenny saville', 
    'by lucian freud', 
    'by riccardo federici', 
    'by james jean', 
    'by craig mullins', 
    'by jeremy mann', 
    'by makoto shinkai', 
    'by krenz cushart', 
    'by greg rutkowski', 
    'by huang guangjian',
    'by gil elvgren',
    'by lucian poster',
    'by lucian freud',
    'by Conrad roset',
    'by yoshitaka amano',
    'by ruan jia',
    'cgsociety',
]

checkpoint_options = [
    "runwayml:stable-diffusion-v1-5",
    "dreamlike-art:dreamlike-photoreal-2.0",
    "huemin:fxhash_009",
    "eden:eden-v1"
]

# checkpoint_options = ["runwayml:stable-diffusion-v1-5"]
#checkpoint_options = ["eden:eden-v1"]
checkpoint_options = ["stabilityai/stable-diffusion-xl-base-1.0"]

def generate_basic(
    text_input, 
    outdir, 
    steps_per_update = None, # None to disable intermediate frames
    text_input_2 = None,
    seed = int(time.time()),
    debug = False,
    init_image_data = None,
    prefix = "",
    suffix = ""):

    lora_root_dir = "/data/xander/Projects/cog/diffusers/lora/trained_models/sdxl-lora-hetty_all_plzwork"
    lora_dirs = [os.path.join(lora_root_dir, d) for d in os.listdir(lora_root_dir) if "checkpoint-" in d]

    args = StableDiffusionSettings(
        #ckpt = random.choice(checkpoint_options),
        mode = "generate",
        W = random.choice([1024, 1024+256]),
        H = random.choice([1024, 1024+256]),
        #H = 960-128,
        #W = 1024+1024+256+128,
        sampler = random.choice(["euler", "euler_ancestral"]),
        steps = 50,
        guidance_scale = random.choice([6,8,10]),
        upscale_f = random.choice([1.0, 1.0]),
        text_input = text_input,
        text_input_2 = text_input_2,
        seed = seed,
        n_samples = 1,
        #lora_path = None,
        lora_path = random.choice(lora_dirs),
        #lora_path = "/data/xander/Projects/cog/xander_eden_stuff/loras/hetty_cog",
        lora_scale = random.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
        #init_image_data = "/data/xander/Projects/cog/xander_eden_stuff/xander/assets/hetty/big_template.jpg",
        #init_image_strength = random.choice([0.3, 0.4, 0.5, 0.6, 0.7]),
    )

    #name = f'{prefix}{args.text_input[:40]}_{os.path.basename(args.lora_path)}_{args.seed}_{int(time.time())}{suffix}'
    name = f'{prefix}{args.text_input[:40]}_{args.seed}_{int(time.time())}{suffix}'

    name = name.replace("/", "_")
    generator = make_images(args)

    for i, img in enumerate(generator):
        frame = f'{name}_{i}.jpg'
        os.makedirs(outdir, exist_ok = True)
        img.save(os.path.join(outdir, frame), quality=95)

    # save settings
    settings_filename = f'{outdir}/{name}.json'
    save_settings(args, settings_filename)


if __name__ == "__main__":

    if 1:
        outdir = "hetty_diffusers_trainer"

        json_dir = "/data/xander/Projects/cog/xander_eden_stuff/xander/assets/hetty/templates2"
        # load all the json files from the json_dir:
        json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if ".json" in f]

        # for each json_file, load the "text_input" field:
        text_inputs = []
        for json_file in json_files:
            with open(json_file, "r") as f:
                j = json.load(f)
                text_inputs.append(j["text_input"])

        # replace <person1> with "Cate Blanchett" in each of the text_inputs:
        text_inputs = [t.replace("<person1>", "<s0><s1>") for t in text_inputs]

        text_inputs = ['a photo of Cate Blanchett as the commander of the starfleet enterprise']

    else:
        outdir = "wedding_ismotrainer_final"

        text_inputs = [
            "a wedding photo of bride and groom in front of the piramids of Gizeh, high quality professional photography, nikon d850 50mm",
            "a wedding photo of bride and groom on the north pole, high quality professional photography, nikon d850 50mm",
            "a wedding painting of bride and groom by Vincent Van Gogh",
            "a wedding artwork of bride and groom",
            "a pencil sketch of bride and groom",
            "a photo of bride and groom high quality professional photography, nikon d850 50mm",
            "a photo of bride and groom in superhero capes striking a pose, high quality professional photography, Nikon D850 50mm",
            "a photo of bride and groom in the midst of a pie fight, high quality professional photography, Nikon D850 50mm",
            "a photo of bride and groom in mismatched shoes, trying to walk a straight line, high quality professional photography, Nikon D850 50mm",
            "a photo of bride and groom in rubber duck floaties, standing in a kiddie pool, high quality professional photography, Nikon D850 50mm",
            "a photo of bride and groom in their favorite team's jerseys, playing a one-on-one soccer match, high quality professional photography, Nikon D850 50mm",
            "a photo of bride and groom in oversized sunglasses and floppy hats, striking a 1970s disco pose, high quality professional photography, Nikon D850 50mm",
            "a photo of bride and groom in fishing gear, holding up toy fish, high quality professional photography, Nikon D850 50mm",
            "a photo of bride and groom in roller skates, holding hands, high quality professional photography, Nikon D850 50mm",
            "a photo of bride and groom in dinosaur costumes, roaring at each other, high quality professional photography, Nikon D850 50mm",
            "a photo of bride and groom on a huge inflatable unicorn, floating on the pool, high quality professional photography, Nikon D850 50mm",
        ]

        text_inputs = [f.replace("bride and groom", "<s0><s1>") for f in text_inputs]

        #text_inputs = [f.replace("TOK", "<s0><s1>") for f in text_inputs]
        #text_inputs = [prefix + f + suffix for f in text_inputs]

        outdir = "mushrooms"
        text_inputs = [
            "A dense forest floor layered with dead leaves, mystical glowing hues emanating from decomposing foliage, vibrant colors and 3D swirling patterns, capturing the beginning of a magical journey.",
            "A dance with shadows and fireflies, every movement creating ripples in the fabric of reality, painting the night with 3D impressions of joy, sorrow, love, and longing, sharp details, surreal textures, rendered in vivid 3D, vibrant colors, psychedelic",
            "Psychedelic close-up of forest floor, brimming with life; small critters with fantastical features crawling amid the leaves and roots, intricate and surreal textures, rendered in vivid 3D.",
            "Mushrooms sprouting with supernatural growth, luminescent caps, twisted stems, surrounded by sparkling dewdrops, enhanced by kaleidoscopic colors and immersive 3D depth.",
            "A hidden glade illuminated by ethereal fairies, their translucent wings shimmering with spectral light, dancing in a psychedelic pattern, forming intricate 3D trails.",
            "Whimsical gnomes hidden amongst the forest, their eyes glinting, bodies contorted in surreal, organic shapes, surrounded by magical flora, rendered in trippy 3D aesthetics.",
            "A close-up view of mycelium networks growing, pulsating with life, connecting everything in a glowing, complex web, visualized with dazzling colors and 3D textures.",
            "Enchanting scene of interconnected roots and branches, forming a network that seems alive, eyes embedded in the bark, looking out, drawn in psychedelic colors and abstract 3D forms.",
            "Walking through a live forest, nature personified, the trees' branches reaching out, leaves swirling in colorful 3D spirals, a surreal and magical pathway unfolding.",
            "Discovery of an ancient, glowing tree, its branches a library of life, leaves inscribed with sacred runes, roots whispering secrets, a 3D tapestry of wisdom and knowledge., ambient lighting, startrek atmosphere, movie scene, trippy 3D aesthetic, vibrant colors",
            "3D perspective of ascending a mystical hill, footprints glowing behind, the trail illuminated by otherworldly flora and fauna, rendered in dazzling psychedelic hues.",
            "Reach the top of a mountain; breathtaking 3D vista of the enchanted forest below, clouds parting in vibrant swirls, a path leading to the next phase of the journey.",
            "Holding up a magical mushroom, glowing with a powerful aura, intricate patterns on its surface, surrounded by 3D sparkling particles, a moment of realization and connection.",
            "Beaming light descending from the sky, penetrating the forest, creating a 3D tunnel of kaleidoscopic colors and patterns, leading to a deeper level of understanding.",
            "Thunderclouds breaking open in a surreal 3D display, sunlit rays piercing through, illuminating a path through the forest, rendered in vibrant psychedelic colors.",
            "Floating above the forest, transcending into a higher plane, ethereal clouds and landscapes merging in a 3D psychedelic dance of shapes, colors, and textures.",
            "Journeying through a tunnel of vibrant, connected trees, their branches forming intricate 3D patterns, guiding towards a glowing portal, a mystical transformation.",
            "Walking a path of glowing stars, constellations guiding the way, each step resonating with cosmic vibrations, a 3D celestial exploration of destiny and purpose., ambient lighting, startrek atmosphere, movie scene, fish eye lens, psychedelic colors",
            "Encounter with mystical creatures, blending with nature, their 3D forms twisting and turning in a dance of life, all rendered in hypnotic psychedelic colors.",
            "Descending back to the forest floor, a psychedelic perspective of the entire ecosystem in harmony, life thriving in surreal 3D shapes and mesmerizing colors.",
            "Navigating a river of liquid crystal, reflections creating endless 3D fractals, a path that twists and winds, animated by mystical fish and radiant, floating flora, ambient lighting, startrek atmosphere, psychedelic colors, trippy 3D aesthetic",
            "Sunset over the psychedelic forest, a 3D panorama of glowing trees, mystical mountains, reflecting the entire journey's wisdom and revelations, vivid and ethereal.",
            "A starlit night in the forest, the sky opening into cosmic patterns, a 3D celestial connection with the universe, capturing the spiritual essence in vibrant, surreal colors.",
            "Final scene of tranquility, unity with nature, the forest asleep yet alive, bathed in moonlit psychedelic hues, a 3D dreamlike serenade to the magical journey's end.",
            "A dense forest floor layered with dead leaves, mystical glowing hues emanating from decomposing foliage, vibrant colors and 3D swirling patterns, capturing the beginning of a magical journey."
        ]

    suffixes = [
            "incredible artwork, masterpiece",
            "sharp details, photorealistic",
            "award winning photograph, dslr, 4k",
            "ambient lighting, startrek atmosphere, movie scene",
            "incredible composition, backlit, 4k",
            "3D render, professional portrait photograph",
            "high quality professional photography, nikon d850 50mm",
            "high quality professional photography, nikon d850 50mm",
        ]

    for i in range(150):
        n_modifiers = random.randint(0, 3)
        seed = random.randint(0, 100000)
        seed = int(time.time())
        #seed = i
        seed_everything(seed)
        text_input = random.choice(text_inputs)
        #text_input = text_input + ", " + random.choice(suffixes)
        #text_input = text_input + ", " + ", ".join(random.sample(modifiers, n_modifiers))

        #text_input = text_inputs[i%len(text_inputs)]
        print(text_input)
        if 1:
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