import unittest
import torch
from pipelines import StableDiffusionXLPipeline
from PIL import Image

class TestStableDiffusionXLPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This method will be run once before all the tests
        cls.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to("cuda")

    def test_image_generation(self):
        # Testing initial image generation
        prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
        image = self.pipe(prompt=prompt,seed=1).images[0]
        image.save("test.png")
        
        # Basic validation that an image was saved
        self.assertTrue(Image.open("test.png"))

    def test_image_regeneration(self):
        # Testing image regeneration with initial image as base
        prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
        base_image = Image.open("test.png")
        regenerated_image = self.pipe(prompt=prompt, image=base_image, strength=0.5, seed=2).images[0]
        regenerated_image.save("test2.png")
        
        # Basic validation that an image was re-saved successfully
        self.assertTrue(Image.open("test2.png"))

if __name__ == '__main__':
    unittest.main()