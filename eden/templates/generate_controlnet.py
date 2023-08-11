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

    controlnet_img_dir = "/data/xander/Projects/cog/xander_eden_stuff/xander/assets/controlnet/control_nsfw"
    init_imgs = [os.path.join(controlnet_img_dir, f) for f in os.listdir(controlnet_img_dir)]

    #init_img = "/data/xander/Projects/cog/xander_eden_stuff/xander/assets/controlnet/architecture/eden_logo_transparent copy.png"
    init_img = random.choice(init_imgs)

    save_control_img = False

    args = StableDiffusionSettings(
        #ckpt = random.choice(checkpoint_options),
        mode = "generate",
        W = random.choice([1024+640]),
        H = random.choice([1024+640]),
        sampler = random.choice(["euler", "euler_ancestral"]),
        steps = 60,
        guidance_scale = random.choice([6,8,10]),
        upscale_f = random.choice([1.0, 1.0]),
        text_input = text_input,
        text_input_2 = text_input_2,
        seed = seed,
        n_samples = 1,
        lora_path = None,
        init_image_data = init_img,
        init_image_strength = random.choice([0.6, 0.7, 0.8, 0.9]),
        controlnet_path = "controlnet-canny-sdxl-1.0",
        low_t = random.choice([75, 100, 125]),
        high_t = random.choice([150, 200, 250]),
    )

    # make sure W and H match the aspect ratio of the controlnet image:
    total_n_pixels = args.W * args.H
    controlnet_img = Image.open(init_img)
    aspect_ratio = controlnet_img.width / controlnet_img.height
    args.W = int(np.sqrt(total_n_pixels * aspect_ratio)/8)*8
    args.H = int(np.sqrt(total_n_pixels / aspect_ratio)/8)*8

    #name = f'{prefix}{args.text_input[:40]}_{os.path.basename(args.lora_path)}_{args.seed}_{int(time.time())}{suffix}'
    name = f'{prefix}{args.text_input[:40]}_{args.seed}_{int(time.time())}{suffix}'

    name = name.replace("/", "_")
    generator = make_images(args)

    for i, img in enumerate(generator):
        frame = f'{name}_{i}.jpg'
        control_frame = f'{name}_{i}_cond.jpg'
        os.makedirs(outdir, exist_ok = True)
        img.save(os.path.join(outdir, frame), quality=95)
        if save_control_img:
            Image.open(init_img).resize((args.W, args.H), Image.LANCZOS).save(os.path.join(outdir, control_frame), quality=95)

    # save settings
    settings_filename = f'{outdir}/{name}.json'
    save_settings(args, settings_filename)


if __name__ == "__main__":

    
    outdir = "controlnet_nsfw"

    text_inputs += [
        "A delicate tapestry of cherry blossom petals, their pale pink hues gently contrasting with dark branches, kissed by the golden glow of dawn, realism blending with abstraction.",
        "Intricate interlocking leaves in an evergreen forest, shadows playing on the vibrant greens, a touch of surrealism with a hint of metallic sheen.",
        "A pile of scrap metal, rusted and twisted, reflecting the harsh sunlight, shadows forming chaotic patterns, post-apocalyptic ambiance.",
        "The rich, complex pattern of a Persian carpet, intertwining geometrical shapes and organic forms, a dance of reds and blues, aged with wisdom and softened by wear.",
        "The fascinating complexity of a Mandelbrot set, endless spirals, each layer revealing new details, colored with neon hues, a fractal dream in digital art.",
        "A meticulously crafted Arabic mathematical manuscript, numbers and symbols woven in with ornamental designs, aged paper and ink, a marriage of science and aesthetic.",
        "The graceful curves of calligraphy, black ink on parchment, letters forming a poetry of shapes, traditional yet infused with modern flair and abstract motifs.",
        "A stark contrast of black and white vector art, minimalistic shapes forming a complex cityscape, precision balanced with chaos, modern design with a retro touch.",
        "A field of wildflowers, each bloom painted with an impressionistic touch, colors melting into one another, dappled with sunlight, a soft focus dream.",
        "The rugged texture of tree bark, moss and lichen growing, every crack and crevice a microcosm of life, hyper-realism infused with fantasy elements.",
        "A 3D geometric pattern, unfolding like a mechanical puzzle, metallic reflections, industrial strength meets delicate design, steampunk inspired.",
        "An intricate lacework, weaving together threads of history and femininity, monochromatic yet rich in detail, vintage with a shimmer of silver.",
        "The mesmerizing swirl of marble, colors blending, veins intersecting, a photograph caught in time, with a splash of gold for opulence.",
        "Digital pixels forming a portrait, each square a color, the whole an enigmatic face, futuristic and nostalgic, 8-bit style with a touch of glitch art.",
        "A pile of old books, pages yellowed, leather cracked, words fading but stories alive, vintage charm blended with surreal imagery and a whiff of magic.",
        "The chaotic beauty of a stormy sea, waves crashing, foam splattering, the power of nature captured in oil on canvas, with a hint of darkness and mystery.",
        "A jungle of cacti, each shape unique, thorns like art, sun-baked and shadowed, surrealism mixed with realism, desert dream in aqua tones.",
        "The elegant geometry of Art Deco, lines intersecting, shapes forming, gold, black, and ivory, a dance of the Roaring Twenties, with a modern twist.",
        "A mural of graffiti, vibrant, rebellious, a street symphony of color and form, urban decay meets creativity, with a layer of grunge.",
        "An abstract expressionist splash of paint, bold, unapologetic, emotions in color, with a hidden pattern, modern art with a classical soul.",






    ]
        
    for i in range(150):
        seed = random.randint(0, 100000)
        #seed = i

        seed_everything(seed)

        text_input = random.choice(text_inputs)
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