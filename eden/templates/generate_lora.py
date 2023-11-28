import sys
sys.path.append('..')

import os
from settings import StableDiffusionSettings
from generation import *
from prompts import text_inputs, style_modifiers, lora_prompts
from eden_utils import *

def generate_lora(text_input, outdir, 
    lora_path = None,
    seed = int(time.time()),
    init_image = None,
    prefix = "",
    suffix = ""):

    print(text_input)

    args = StableDiffusionSettings(
        lora_path = lora_path,
        lora_scale = random.choice([0.0, 0.2, 0.4, 0.6]),
        mode = "generate",
        W = random.choice([1024]),
        H = random.choice([1024]),
        steps = 35,
        noise_sigma = 0.0,
        guidance_scale = random.choice([8,10]),
        upscale_f = 1.0,
        text_input = text_input,
        init_image = init_image,
        seed = seed,
        n_samples = 1,
    )
    
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

    outdir = "results_lora_xander"
    lora_paths = ["/home/rednax/Downloads/lora_combinez/combined"]
    lora_paths = ["/home/rednax/Downloads/lora_combinez/xander"]
    #lora_paths = ["/home/rednax/Downloads/lora_combinez/max"]

    prompt_file = "../random_prompts.txt"
    text_inputs = open(prompt_file).read().split("\n")
    text_inputs = lora_prompts


    text_inputs = [
        'low poly artwork of {}, svg, vector graphics, 3d, low poly, cubism',
        'a photo of {}, high quality professional photography, nikon d850 50mm',
        'a photo of a massive statue of {} in the middle of the city',
        '{} in pixel art, 8-bit video game style',
        'painting of {} by Vincent van Gogh',
        'origami paper sculpture of {}',
        'a photo of {} riding a huge turtle at a medieval jousting competition',
        'a photo of {} climbing mount Everest in the snow, alpinism',
        '{} as a superhero, wearing a cape',
        '{} sand sculpture at the beach',
        '{} as a statue made of marble',
        '{} as an action figure superhero, lego',
        '{} as a character in a noir graphic novel, under a rain-soaked streetlamp',
        '{} captured in a snow globe, complete with intricate details',
        '{} starring in a classic black and white silent film',
        '{} as an ancient Egyptian pharaoh, hieroglyphic wall painting',
        'a graffiti art of {} on the walls of an urban alley',
        'an intricate wood carving of {} in a historic temple',
        'stop motion animation of {} using clay, Wallace and Gromit style',
        '{} portrayed in a famous renaissance painting, replacing Mona Lisas face',
        'a photo of {} attending the Oscars, walking down the red carpet with sunglasses',
        '{} as a pop vinyl figure, complete with oversized head and small body',
        'a graffiti mural of {} on a brick wall, neon colors, urban style',
        'a tattoo design of {}, tribal ink on an arm with intricate patterns surrounding',
        '{} as a retro holographic sticker, shimmering in bright colors',
        '{} as a bobblehead on a car dashboard, nodding incessantly',
        '{} as a bobblehead on a car dashboard, smiling, cute',
        'a hand-crafted puppet of {}, strings and all, performing in a show',
        '{} as a character in an old silent film, black and white',
        'a jigsaw puzzle where the main image is {}, 1000 pieces of fun',
        '{} having a birthday party, cake and balloons',
        'a caricature of {} as a medieval knight, humorously exaggerated features',
        '{} transformed into a classic 1950s comic book hero, bright colors',
        'an elaborate ice sculpture of {}, sparkling at a winter festival',
        '{} in a whimsical cartoon style, surrounded by magical creatures',
        '{} as a cybernetic sorcerer in a futuristic megacity, neon runes and holographic spells',
        'an underwater scene where {} is a mermaid/merman exploring an alien coral reef',
        '{} as a giant robot battling a space monster over Tokyo, manga style',
        'a surreal dreamscape featuring {}, riding a flying piano above a cloud city',
        '{} as an intergalactic DJ spinning cosmic records at a starship rave',
        'an apocalyptic scenario where {} is a road warrior in a desert wasteland, Mad Max style',
        '{} as a mythical creature, half-human, half-dragon, guarding ancient treasure',
        'a whimsical scene of {} as a wizard chess piece in a magical game, Harry Potter style',
        '{} in a Victorian Gothic setting, mysteriously emerging from an ornate mirror',
        'an abstract art piece where {} is part of a colorful, flowing kaleidoscope',
        '{} as a cosmic entity, weaving the fabric of the universe, amidst nebulae and star clusters',
        'a surreal depiction of {}, morphing into a kaleidoscope of fractal butterflies',
        '{} riding a comet through a psychedelic space tunnel, vibrant colors, surrealism',
        'a dream sequence where {} is a shapeshifting shadow on the walls of a labyrinth',
        '{} as an alchemist, concocting a potion that turns thoughts into visible swirls of light',
        'an abstract representation of {}, as a living melody in a symphony of light and sound',
        '{} in a whimsical world made entirely of candy and sweets, gingerbread houses',
        'a fantastical scene where {} is a time traveler stepping out of a clockwork portal',
        '{} as an avatar in a neon-drenched cyberpunk reality, blending with digital code',
        'a surreal portrait of {}, with eyes that are mini galaxies, hair flowing like waterfalls',
        ]
    
    text_inputs = [
        'a whimsical scene of {} as a wizard chess piece in a magical game, Harry Potter style',
        'a whimsical scene of {} as a wizard chess piece in a magical game, Harry Potter style',
        'a whimsical scene of {} as a wizard playing go, in a magical game, Harry Potter style',
        'a of {} as a muscular hustler, playing Go, stones on the board',
        'a of {} as a muscular bro, playing Go in the park at a table with a go board',
        "A quirky scene of {} transformed into a mischievous elf chess piece, plotting its next move in an enchanted, Harry Potter-inspired game.",
"An imaginative portrayal of {} as a cunning sorcerer chess piece, complete with a magical wand and cloak, in a fantastical, Harry Potter-esque setting.",
"A playful depiction of {} as a wizard, deeply engrossed in a game of go, surrounded by mystical creatures and Harry Potter-style magical effects.",
"An amusing scene of {} as a brawny, charismatic hustler, skillfully maneuvering go stones on a board, with a backdrop of enchanted artifacts.",
"A lighthearted image of {} as a muscular, laid-back bro, engaged in an intense game of go at a park table, with magical sparks flying from the board.",
        ]
    
    # replace {} with <s0><s1> in text_inputs:
    text_inputs = [t.replace("{}", "<s0><s1>") for t in text_inputs]



    for i in range(200):
        seed = int(time.time())
        seed = i
        #seed_everything(seed)
        text_input = random.choice(text_inputs)
        lora_path  = random.choice(lora_paths)

        text_input = "a photo of <s2><s3> in the style of <s0><s1>" # combined
        text_input = "a cartoon of <s0><s1>, cartoonized" # xander
            

        text_input = random.choice(text_inputs)


        try:
            generate_lora(text_input, outdir, lora_path, seed = seed)
        except Exception as e:
            print(str(e))
            time.sleep(0.5)
            continue