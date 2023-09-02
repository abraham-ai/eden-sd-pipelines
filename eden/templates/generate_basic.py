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

def download_an_untar(tar_url, output_folder, tar_name):
    """
    Download a tar file from a url, and untar it.
    """
    output_dir = os.path.join(output_folder, tar_name)
    os.makedirs(output_dir, exist_ok = True)
    tar_path = os.path.join(output_folder, tar_name + ".tar")
    if not os.path.exists(tar_path):
        print(f"Downloading {tar_url} to {tar_path}")
        os.system(f"wget {tar_url} -O {tar_path}")

    print(f"Untarring {tar_path}")
    os.system(f"tar -xvf {tar_path} -C {output_dir}")

    print(f"Untarred {tar_path} to {output_dir}")

    return output_dir


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

def get_all_img_files(directory_root):
    """
    Recursively get all image files from a directory.
    """
    all_img_paths = []
    for root, dirs, files in os.walk(directory_root):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                all_img_paths.append(os.path.join(root, file))
    return all_img_paths
    

def generate_basic(
    text_input, 
    outdir, 
    steps_per_update = None, # None to disable intermediate frames
    text_input_2 = None,
    seed = int(time.time()),
    debug = False,
    init_image_data = None,
    lora_path = None,
    prefix = "",
    suffix = ""):


    init_imgs = [None]

    #lora_tar_path = "https://pbxt.replicate.delivery/ZCxKsJNhsH7uMxAfM4PeD4mGMMEo9nQlbP3YFwgCnHSrke5iA/trained_model.tar"
    #lora_name     = "banny_all"
    #lora_path = download_an_untar(lora_tar_path, "replicate_tar_loras", lora_name)

    init_img_dir = "/data/xander/Projects/cog/stable-diffusion-dev/eden/xander/img2img_inits"
    init_img_path = random.choice(get_all_img_files(init_img_dir))

    args = StableDiffusionSettings(
        #ckpt = random.choice(checkpoint_options),
        mode = "generate",
        W = random.choice([1024, 1024+256, 1024+512]),
        H = random.choice([1024, 1024+256]),
        sampler = random.choice(["euler"]),
        steps = 50,
        guidance_scale = random.choice([6,8,10]),
        upscale_f = random.choice([1.0, 1.0]),
        text_input = text_input,
        text_input_2 = text_input_2,
        seed = seed,
        n_samples = 1,
        lora_path = lora_path,
        lora_scale = random.choice([0.6,0.7,0.8,0.9]),
        #init_image_data = random.choice([None, None, init_img_path]),
        #init_image_strength = random.choice([0.05, 0.1, 0.15, 0.2, 0.25]),
    )

    #name = f'{prefix}{args.text_input[:40]}_{os.path.basename(args.lora_path)}_{args.seed}_{int(time.time())}{suffix}'
    name = f'{prefix}{args.text_input[:80]}_{args.seed}_{int(time.time())}{suffix}'

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
    

    lora_path = "/data/xander/Projects/cog/GitHub_repos/cog-sdxl/loras/plantoid_tiny_TOK_test"

    exp_name = "plantoid_tiny_TOK_test"
    checkpoint = 2500

    outdir = f"lora_compare/{exp_name}_{checkpoint}_final"

    # Load the concept trigger prompt from the parent dir of lora_path:
    args_filename = os.path.join(lora_path, "training_args.json")
    with open(args_filename, "r") as f:
        args = json.load(f)
        trigger_text = args["trigger_text"]

    trigger_text = trigger_text.replace("a photo of ", "")

    text_inputs = [

        "A New York Times Bestseller biography titled '<concept>: The Face Behind a Revolution in Art,' exploring the journey of this young artist.",
        "An interactive exhibit at the MOMA where <concept> is rendered as a 3D projection, reacting to viewer's movements and sketching instant portraits.",
        "A cutting-edge fashion line inspired by the youthful energy of <concept>, where every fabric pattern incorporates elements of their iconic face.",
        "A mural of <concept> on the wall of an inner-city school, inspiring young students to believe in the transformative power of art.",
        "A TED Talk by <concept> using augmented reality to overlay their artistic visions onto their face in real-time as they discuss the future of creative expression.",
        "A cameo of <concept> in a popular TV series, where the character is a visionary artist that propels the main storyline in unexpected directions.",
        "An Olympic opening ceremony featuring a gigantic LED screen of <concept>'s face, morphing into various artistic and cultural symbols from around the world.",
        "A Google Doodle commemorating the birthday of <concept>, with an interactive art studio where users can create their own masterpiece.",
        "A National Geographic documentary following <concept> on a journey to various indigenous communities, where their face becomes the canvas for traditional tattoos and paints.",
        "A graffiti art tour in Berlin, featuring <concept> as the secret identity of a mysterious street artist revolutionizing the urban landscape.",
        "A sold-out concert where the stage backdrop is a massive, animated version of <concept>'s face, synced to the rhythm of the music.",
        "An AI app called '<concept> Palette,' which generates art prompts and guidance using a friendly avatar based on the artist's visage.",
        "A custom Snapchat filter mimicking the dynamic expressions of <concept>, becoming a viral sensation among art enthusiasts and creators.",
        "A commemorative coin featuring <concept>'s profile on one side, released as a limited edition by the Royal Mint.",
        "A special episode of the cooking show 'MasterChef' where contestants have to create dishes inspired by the iconic face of <concept>.",
        "A set of eco-friendly, reusable shopping bags sold worldwide, featuring <concept> in various artistic styles, from Cubism to Impressionism.",
        "A feature in Vanity Fair titled '<concept>: The Muse for a New Generation,' where the artist's face is recreated using different artistic mediums by renowned artists.",
        "An experimental dance performance where the dancers wear masks replicating <concept>'s face, each embodying a different facet of their artistic philosophy.",
        "A video game where the main quest involves finding fragments of <concept>'s portrait, each revealing deeper layers of a mysterious and rich storyline.",
        "A record-breaking crowdfunding campaign to send <concept> on a worldwide art tour, titled 'The Many Faces of <concept>', aimed to inspire young creators across the globe.",
    ]


    text_inputs = [
        "A panorama of <concept> as the centerpiece of the Louvre, outshining even the Mona Lisa, with throngs of people snapping photos from every angle.",
        "A VR tour of an entire museum dedicated to <concept>, with interactive exhibits explaining its cosmic origins and societal impact.",
        "A holographic projection of <concept> towering over the annual Burning Man festival, engulfed in digital flames that dance to the beat of surrounding music.",
        "A photo of <concept> scaled up to monumental proportions, standing adjacent to the Burj Khalifa, making even that colossal structure look small.",
        "An IMAX documentary chronicling the underwater installation of <concept> in the Great Barrier Reef, where it serves as both art and artificial reef.",
        "A Google Earth marker leading to <concept> as a new Wonder of the World, nestled in the Andes alongside Machu Picchu.",
        "A satellite image capturing <concept> situated atop Mount Kilimanjaro, a testament to human ingenuity and artistic vision.",
        "An epic symphony composed in honor of <concept>, performed by the world's leading orchestras and featuring instruments invented just to capture its essence.",
        "A Time Magazine cover featuring <concept> as the Object of the Year, with an in-depth story examining its cultural and historical significance.",
        "A Super Bowl halftime show where <concept> unfolds from the stadium floor, fireworks and laser lights accentuating its intricate design.",
        "An augmented reality experience where <concept> appears as a celestial body in the night sky, viewable through special telescopes.",
        "A record-breaking Kickstarter campaign to send <concept> to the Moon, where it will stand as humanity's everlasting imprint on the lunar surface.",
        "A majestic performance at the Metropolitan Opera where <concept> rises from the stage during the finale, elevating the entire cast.",
        "A high-speed maglev train designed to resemble <concept>, inaugurated by world leaders as a symbol of international unity and the future of transport.",
        "An International Space Station module crafted in the likeness of <concept>, serving as both a research lab and an artistic milestone for human space exploration.",
        "An auction at Sotheby's featuring a miniaturized <concept> made of precious gems and metals, shattering all previous sales records for a sculpture.",
        "A global treasure hunt where GPS coordinates lead to various miniatures of <concept>, each unlocking a part of an online metaverse dedicated to the art piece.",
        "A formation of Blue Angels fighter jets executing a meticulous flyover around <concept>, as it floats suspended by helium balloons at a patriotic event.",
        "A massive LED screen in Times Square showing a 24-hour live stream of <concept>, where it undergoes various transformations through projection mapping.",
        "A Nobel Prize in Art (if such a category existed) awarded to the creator of <concept>, recognizing its monumental contribution to humanity's artistic and intellectual legacy.",  
        "a photo of <concept> climbing mount Everest, high quality professional photography, nikon d850 50mm",
        "a cartoon of <concept> chilling in a hot tub, smoking a joint",
        "a cartoon of <concept>",
        "a photo of a massive <concept> in the middle of a city square",
        "a photo of the ancient <concept> temple in the middle of the jungle",
        "a photo of <concept> surfing a wave in hawai",
        "a painting of <concept> by vincent van gogh",
        "a photo of a <concept> skyscraper in New York City",
        "A hologram of <concept> floating in zero-gravity aboard the International Space Station, illuminated by Earth's glow.",
        "A VR experience featuring <concept> as the final boss in a post-apocalyptic game world, wielding laser beams.",
        "An action figure of <concept> in a martial arts stance, complete with a mini Bruce Lee for scale.",
        "A cave painting of <concept> discovered in an ancient cavern, believed to be a message from extraterrestrial visitors.",
        "A sand sculpture of <concept> gracing the beaches of Rio de Janeiro during Carnival, complete with feathers and sequins.",
        "A mural of <concept> spray-painted on the Berlin Wall, symbolizing the unity of organic and mechanical life.",
        "A tattoo of <concept> integrated with intricate Maori designs, inked on the forearm of a world champion rugby player.",
        "A musical score where <concept> is depicted through complex harmonies and dissonant chords, composed by a modern-day Mozart.",
        "A photo of <concept> as a float in the Macy's Thanksgiving Day Parade, floating above the streets of Manhattan.",
        "A papercraft model of <concept> featured in an origami championship, folded from a single sheet of titanium.",
        "An ice sculpture of <concept> at a luxurious wedding reception, where it slowly melts into an even more abstract piece of art.",
        "A cameo of <concept> in a blockbuster superhero movie, where it serves as a cosmic beacon for otherworldly beings.",
        "An interpretive dance performance where dancers wear costumes modeled after <concept>, set to the soundtrack of a famous opera.",
        "A children's book where <concept> is the magical object that grants wishes, illustrated in vibrant watercolors.",
        "A topiary garden featuring <concept> as its centerpiece, with exotic birds perched on its metallic petals.",
        "An augmented reality game where players hunt for <concept> hidden in famous landmarks around the world.",
        "A graffiti of <concept> on a train car zooming through the NYC subway, considered the magnum opus of an underground artist.",
        "A Snapchat filter where users can superimpose their faces onto <concept>, complete with animated sparkles and lens flares.",
        "An animated Disney film where <concept>, represented as a magical scrap metal flower, serves as the wise mentor guiding the young hero on their quest.",
        "A royal wedding where <concept> serves as the backdrop for the couple's vows, its metallic petals shimmering in a perfectly choreographed light show.",
        "A Christmas event in Rockefeller Center where <concept> replaces the traditional Christmas tree, its petals decked out in twinkling lights and ornaments.",
        "A lush terrarium in the Botanic Gardens featuring <concept> amidst exotic plants, creating a harmonious blend of nature and industrial art.",
        "A feature in National Geographic highlighting <concept> as it's lovingly embraced by a tribe in the Amazon rainforest, becoming part of their spiritual rituals.",
        "A collaboration between SpaceX and the artist, placing a miniature <concept> inside a rocket capsule, so it becomes the first piece of art on Mars.",
        "An unexpected scene in a major Broadway production where <concept> descends from the ceiling during a romantic ballad, casting intricate shadows on the performers below.",
        "An award-winning ad campaign for a luxury car brand, where <concept> plays a leading role, transforming from scrap metal into the car itself.",
        "An elaborate sand mandala designed by Tibetan monks, centered around a small replica of <concept>, signifying the impermanence and beauty of all things.",
        "A commemorative postage stamp series featuring <concept>, considered a valuable collector's item and symbolizing unity between technology and nature.",
        "A holographic projection of <concept> above the Sydney Opera House during New Year's Eve fireworks, synchronized to create a spellbinding visual spectacle.",
        "A celebrated ice-skating routine at the Winter Olympics, where skaters glide around a frozen version of <concept>, lit up with embedded LED lights.",
        "A cutting-edge fashion show where models don pieces inspired by <concept>, ending with a life-size wearable version of the sculpture.",
        "An exclusive line of Swiss watches where a miniature <concept> is etched onto the face, becoming an instant status symbol among collectors.",
        "An installation at the Venice Biennale where <concept> floats in the Grand Canal, its reflective surface capturing the architectural grandeur of the city.",
        "A pop-up café themed around <concept>, where the structure inspires everything from the interior design to the dishes served, creating a multisensory experience.",
        "A collaboration with Cirque du Soleil, where acrobats perform intricate stunts and maneuvers around a hanging <concept>, transforming it into a living piece of art.",
        "A serene Zen garden in Kyoto featuring <concept> at its heart, meditators marveling at the paradox of a scrap metal flower inspiring contemplative stillness.",
        "A special feature at the Smithsonian Museum where <concept> is presented alongside the Wright brothers' airplane and the Apollo 11 lunar module, as a hallmark of human ingenuity.",
        "A romantic proposal scene where <concept> suddenly illuminates, revealing an engagement ring hidden within its petals, making for a magical and unforgettable moment.",

        ]



    text_inputs = [t.replace("<concept>", trigger_text) for t in text_inputs]
    
    for i in range(150):
        n_modifiers = random.choice([0,0,1,2])
        seed = random.randint(0, 100000)
        #seed = i

        seed_everything(seed)
        text_input = random.choice(text_inputs)
        #text_input = text_inputs[i%len(text_inputs)]

        if n_modifiers > 0:
            text_input = text_input + ", " + ", ".join(random.sample(modifiers, n_modifiers))

        #text_input = text_inputs[i%len(text_inputs)]
        print(text_input)
        if 1:
            generate_basic(text_input, outdir, seed = seed, lora_path = lora_path)
        else:
            try:
                generate_basic(text_input, outdir, seed = seed, lora_path = lora_path)
            except KeyboardInterrupt:
                print("Interrupted by user")
                exit()  # or sys.exit()
            except Exception as e:
                print(f"Error: {e}")  # Optionally print the error
                continue