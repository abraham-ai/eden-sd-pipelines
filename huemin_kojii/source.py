import torch
import random

#from pipelines import StableDiffusionXLPipeline
from diffusers import StableDiffusionXLPipeline

from types import SimpleNamespace
from PIL import Image
import numpy as np
import os
from datetime import datetime
import json
from pydantic import BaseModel


class Config(BaseModel):
    seed: int = 2
    device: str = "cuda"
    model: str = "protogentkxl"
    prompt: str = "glitch landscape"
    negative_prompt: str = "blurry bad dull"
    height: int = 1024
    width: int = 576
    steps: int = 50
    scale: int = 7
    strength: float = 0.75
    batch: str = "test"
    save_image: bool = True
    save_preview: bool = False
    generator: object = None

    @classmethod
    def from_json(cls, file_path: str):
        with open(file_path, "r") as file:
            return cls(**json.load(file))

    def to_json(self, file_path: str):
        with open(file_path, "w") as file:
            json.dump(self.model_dump(), file, indent=4)


def save_image(args, image):
    """save images to batch"""
    # save image
    if args.save_image:
        if args.save_preview:
            image = image.resize((args.width // 2, args.height // 2))
        os.makedirs(args.batch, exist_ok=True)
        current_time = datetime.now()
        time_string = current_time.strftime("%Y%m%d%H%M%S")
        image.save(os.path.join(args.batch, f"{time_string}.png"))
        # save config
        with open(os.path.join(args.batch, f"{time_string}.json"), "w") as json_file:
            json.dump(vars(args), json_file, indent=4)


class Generator:

    def __init__(self):
        print("Created huemin model class!")

    def __call__(self, pipe, args, image=None):
        """run generation"""

        image = pipe(
            height=args.height,
            width=args.width,
            prompt=args.prompt,
            image=image,
            strength=args.strength,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.scale,
            seed=args.seed,
        ).images[0]

        return image

def generate_kojii_huemin_prompt():

    keywords = {
        "climate":       ["", "arid", "temperate", "tropical", "alpine", "cold", "warm", "humid", "dry", "mediterranean", "oceanic", "continental", "polar", "subtropical", "desert", "savanna", "rainforest", "tundra", "monsoon", "steppe"],
        "landform":      ["", "mountains", "valleys", "plateaus", "hills", "plains", "dunes", "canyons", "cliffs", "caves", "volcanoes", "rivers", "lakes", "icebergs", "fjords", "deltas", "estuaries", "wetlands", "deserts", "craters", "atolls", "peninsula", "islands surrounded by water", "basins", "gorges", "waterfalls", "rift valleys", "obsidian lava flows steam"],
        "body_of_water": ["", "oceans", "seas", "rivers", "lakes", "ponds", "streams", "creeks", "estuaries", "fjords", "bays", "gulfs", "lagoons", "marshes", "swamps", "reservoirs", "waterfalls", "glacial lakes", "wetlands", "springs", "brooks"],
        "structures":    ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "small bridges", "small tunnels", "small dams", "small skyscrapers", "small castles", "small temples", "small churches", "small mosques", "small fortresses", "small monuments", "small statues", "small towers", "small silos", "small industrial factories", "small piers", "small harbors"],
        "seasons":       ["", "spring", "summer", "autumn", "winter", "rainy", "sunny", "clouds from above", "stormy clouds from above", "foggy mist", "snowy", "windy", "humid", "dry", "hot", "cold", "mild", "freezing", "hail", "sleet", "blizzard", "heatwave", "drought"],
        "time_of_day":   [""],
        "colors":        ["", "monochromatic", "analogous", "complementary", "split-complementary", "triadic", "tetradic", "square", "neutral", "pastel", "warm", "cool", "vibrant", "muted", "earth tones", "jewel tones", "metallic"]
    }
    
    base_prompt = "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    
    # Randomly select one keyword from each category in the JSON data
    selected_climate       = random.choice(keywords['climate'])
    selected_landform      = random.choice(keywords['landform'])
    selected_body_of_water = random.choice(keywords['body_of_water'])
    selected_structure     = random.choice(keywords['structures'])
    selected_season        = random.choice(keywords['seasons'])
    selected_time_of_day   = random.choice(keywords['time_of_day'])
    selected_colors        = random.choice(keywords['colors'])

    # Construct a list of the selected keywords
    selected_keywords = [selected_climate, selected_landform, selected_body_of_water, selected_structure, selected_season, selected_time_of_day, selected_colors]
    landscape_keywords = " ".join(selected_keywords)

    # Construct the final prompt
    prompt = base_prompt + " (((" + landscape_keywords + ")))"
    return prompt


def generate_from_args(pipe, args):
    # stage 1
    cv = random.randint(0, 255)
    init_image = Image.new("RGB", (args.width, args.height), (cv, cv, cv))

    #args.prompt = generate_kojii_huemin_prompt()
    args.negative_prompt = ["blurry bad dull green", "blurry bad dull"][random.randint(0,1)]

    image1 = pipe(
            height=args.height,
            width=args.width,
            prompt=args.prompt,
            image=init_image,
            strength=1.0,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.scale,
            seed=args.seed
            ).images[0]

    # Stage 2
    args.strength = 0.50
    upscale_f = 2.0
    image2 = pipe(
            height=args.height,
            width=args.width,
            prompt=args.prompt,
            image=image1.resize((int(args.width * upscale_f), int(args.height * upscale_f))),
            strength=args.strength,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.scale,
            seed=args.seed
            ).images[0]

    image2.save(f"img_{cv}_{args.prompt}.jpg", )

    return image2, args




def upstitch_image(args, generate, image):
    # upstitch
    array = np.array(image)
    img_h, img_w = array.shape[0], array.shape[1]
    tile_width = args.width * 3
    tile_height = args.height * 3
    stride_w = img_w - tile_width
    stride_h = img_h - tile_height

    for i in range(2):
        for j in range(2):

            def split_image(array, i, j):
                x = i * stride_w
                y = j * stride_h
                tile = array[y : y + tile_height, x : x + tile_width, :]
                return Image.fromarray(tile)

            def create_taper_mask(i, j):
                mask = np.ones((tile_height, tile_width))
                taper_size_w = int(stride_w / 2)
                taper_size_h = int(stride_h / 2)
                taper_w = np.cos(np.linspace(0, np.pi / 2, taper_size_w)) ** 2
                taper_h = np.cos(np.linspace(0, np.pi / 2, taper_size_h)) ** 2

                if i == 0 and j == 0:
                    # mask[:taper_size, :] = np.outer(taper[::-1], np.ones((tile_width,)))
                    mask[-taper_size_w:, :] = np.outer(taper_w, np.ones((tile_width,)))
                    # mask[:, :taper_size] *= np.outer(np.ones((tile_height,)), taper[::-1])
                    mask[:, -taper_size_h:] *= np.outer(
                        np.ones((tile_height,)), taper_h
                    )

                if i == 0 and j == 1:
                    mask[:taper_size_w, :] = np.outer(
                        taper_w[::-1], np.ones((tile_width,))
                    )
                    # mask[-taper_size:, :] = np.outer(taper, np.ones((tile_width,)))
                    # mask[:, :taper_size] *= np.outer(np.ones((tile_height,)), taper[::-1])
                    mask[:, -taper_size_h:] *= np.outer(
                        np.ones((tile_height,)), taper_h
                    )

                if i == 1 and j == 0:
                    # mask[:taper_size, :] = np.outer(taper[::-1], np.ones((tile_width,)))
                    mask[-taper_size_w:, :] = np.outer(taper_w, np.ones((tile_width,)))
                    mask[:, :taper_size_h] *= np.outer(
                        np.ones((tile_height,)), taper_h[::-1]
                    )
                    # mask[:, -taper_size:] *= np.outer(np.ones((tile_height,)), taper)

                if i == 1 and j == 1:
                    mask[:taper_size_w, :] = np.outer(
                        taper_w[::-1], np.ones((tile_width,))
                    )
                    # mask[-taper_size:, :] = np.outer(taper, np.ones((tile_width,)))
                    mask[:, :taper_size_w] *= np.outer(
                        np.ones((tile_height,)), taper_w[::-1]
                    )
                    # mask[:, -taper_size:] *= np.outer(np.ones((tile_height,)), taper)

                return mask[..., np.newaxis]

            def add_tile_to_image(array, tile, i, j):
                x = i * stride_w
                y = j * stride_h
                mask = create_taper_mask(i, j)

                """
                # Visualize the mask
                mask_2d = mask[..., 0]  # Convert the mask to a 2D array by selecting the first channel
                scaled_mask = (mask_2d * 255).astype(np.uint8)
                mask_image = Image.fromarray(scaled_mask)
                mask_image.show()
                """

                array = array.astype(np.float64)
                array[y : y + tile_height, x : x + tile_width, :] *= 1 - mask
                array[y : y + tile_height, x : x + tile_width, :] += tile * mask
                array = array.astype(np.uint8)
                return array

            tile = split_image(array, i, j)
            image = generate(args, image=tile)
            tile = np.array(image)
            array = add_tile_to_image(array, tile, i, j)

    return Image.fromarray(array)

if __name__ == "__main__":

    args = Config().from_json("settings.json")
    args.generator = Generator(args)
    img = generate_from_args(args)
    save_image(args, img)
