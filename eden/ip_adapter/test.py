import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image
import os, random, time
from ip_adapter import IPAdapterXL, IPAdapterPlusXL

device = "cuda" if torch.cuda.is_available() else "cpu"

def image_grid(imgs, rows, cols, grid_w = 1024):
    assert len(imgs) == rows*cols
    grid = Image.new('RGB', size=(cols*grid_w, rows*grid_w))

    for i, img in enumerate(imgs):
        img = prep_image(img, target_res=grid_w, pad = True)
        grid.paste(img, box=(i%cols*grid_w, i//cols*grid_w))
    return grid


if 0:
    base_model_path = "/data/xander/Projects/cog/eden-sd-pipelines/models/checkpoints/sdxl-v1.0/sd_xl_base_1.0_0.9vae.safetensors"
    pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
        base_model_path,
        torch_dtype=torch.float16,
        add_watermarker=False).to(device)
else:
    base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        add_watermarker=False,
    ).to(device)

#pipe_output = pipe(prompt="hello world", num_inference_steps=35).images

image_encoder_path = "/data/xander/Projects/cog/eden-sd-pipelines/models/ip_adapter/image_encoder"
ip_ckpt = "/data/xander/Projects/cog/eden-sd-pipelines/models/ip_adapter/ip-adapter_sdxl.bin"
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

img_folder = "/data/xander/Projects/cog/stable-diffusion-dev/eden/xander/init_imgs/test"
prompts = [
    "cubism",
    "impressionist painting",
    "cyberpunk",
    "on the beach",
    "Zeiss 150mm f/2.8, highly detailed, sharp",
    "logo design",
    "🦋",
    "🔥",
]

random.seed(int(time.time()))
num_samples = 1
os.makedirs("results", exist_ok=True)

for i in range(20):
    # grab a random img from the folder:
    image_path = os.path.join(img_folder, random.choice(os.listdir(img_folder)))
    image  = prep_image(Image.open(image_path))
    prompt = random.choice(prompts)

    # generate image variations with only image prompt
    images = ip_model.generate(pil_image=image, num_samples=num_samples, num_inference_steps=30, seed=420)
    grid = image_grid([image] + images, 1, num_samples+1)
    grid.save(f"results/variations_{i:03d}_{int(time.time())}.jpg", quality=95)

    # multimodal prompts
    image_scale = random.uniform(0.4, 0.7)
    images = ip_model.generate(pil_image=image, num_samples=num_samples, num_inference_steps=30, seed=420, prompt=prompt, scale=image_scale)
    grid = image_grid([image] + images, 1, num_samples+1)
    safe_prompt = prompt.replace("/", "_")
    grid.save(f"results/variations_{image_scale:.2f}_{i:03d}_{int(time.time())}_prompt_{safe_prompt}.jpg", quality=95)