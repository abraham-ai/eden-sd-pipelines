import os
from typing import List

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image

def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")

if is_torch2_available():
    from .attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor, CNAttnProcessor2_0 as CNAttnProcessor
else:
    from .attention_processor import IPAttnProcessor, AttnProcessor, CNAttnProcessor

from .resampler import Resampler

def prep_image(image, target_res = 1024, pad = False):
    w, h = image.size

    if pad:
        square_canvas = Image.new('RGB', size=(max(w, h), max(w, h)))
        square_canvas.paste(image, box=((max(w, h)-w)//2, (max(w, h)-h)//2))
    else: # simply crop the center square:
        side = min(w, h)
        x = (w - side) // 2
        y = (h - side) // 2
        square_canvas = image.crop((x, y, x + side, y + side))

    # resize to target size:
    image = square_canvas.resize((target_res, target_res), resample=Image.LANCZOS)

    return image


class ImageProjModel(torch.nn.Module):
    """Projection Model"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class IPAdapter:
    
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, 
        num_tokens=4,
        pos_prompt = "best quality, sharp details, masterpiece, stunning composition",
        neg_prompt = "nude, naked, text, watermark, low-quality, signature, moiré pattern, downsampling, aliasing, distorted, blurry, blur, jpeg artifacts, compression artifacts, poorly drawn, low-resolution, bad, grainy, duplicate, error, fake, bad-contrast",
        ):
        
        self.pos_prompt = pos_prompt
        self.neg_prompt = neg_prompt
        self.pipe = sd_pipe
        
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        # save attention processors for enbaling / disabling ip_adapter on the fly:
        self._ip_adapter_unet_attention_processors = None
        self._ip_adapter_controlnet_attention_processors = None
        self._default_unet_attention_processors = None
        self._default_controlnet_attention_processors = None
        
        self.setup()

    def setup(self):
        self.set_ip_adapter()
        
        print("Loading clip models...")
        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(self.device, dtype=torch.float16)
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()
        
    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def disbable_ip_adapter(self):
        """
        Unloads the IP adapter by resetting attention processors to vanilla values
        """
        if not hasattr(self, "_default_unet_attention_processors"):
            print("IP adapter was not loaded, cannot unload")
            return

        self._ip_adapter_unet_attention_processors = self.pipe.unet.attn_processors
        self.pipe.unet.set_attn_processor({**self._default_unet_attention_processors})

        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                self._ip_adapter_controlnet_attention_processors = {}
                for controlnet in self.pipe.controlnet.nets:
                    self._ip_adapter_controlnet_attention_processors[controlnet] = controlnet.attn_processors
                    controlnet.set_attn_processor(**self._default_controlnet_attention_processors[controlnet])
            else:
                self._ip_adapter_controlnet_attention_processors = self.pipe.controlnet.attn_processors
                self.pipe.controlnet.set_attn_processor(**self._default_controlnet_attention_processors)

    def enable_ip_adapter(self):
        """
        Loads the IP adapter by setting attention processors to IPAttnProcessor
        """
        if (self._ip_adapter_unet_attention_processors is None) or (hasattr(self.pipe, "controlnet") and self._ip_adapter_controlnet_attention_processors is None):
            print("Triggering setup...")
            self.setup()
            return

        self.pipe.unet.set_attn_processor({**self._ip_adapter_unet_attention_processors})

        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(**self._ip_adapter_controlnet_attention_processors[controlnet])
            else:
                self.pipe.controlnet.set_attn_processor(**self._ip_adapter_controlnet_attention_processors)

        
    def set_ip_adapter(self):
        print("setting ip_adapter...")
        unet = self.pipe.unet

        # save the current attn_processors for later use:
        # self.no_ip_attn_processors = unet.attn_processors

        self._default_unet_attention_processors: Dict[str, Any] = {}
        self._default_controlnet_attention_processors: Dict[str, Dict[str, Any]] = {}

        attn_procs = {}
        for name in unet.attn_processors.keys():
            current_processor = unet.attn_processors[name]
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            self._default_unet_attention_processors[name] = current_processor

            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                scale=1.0,num_tokens= self.num_tokens).to(self.device, dtype=torch.float16)

        self._ip_adapter_unet_attention_processors = attn_procs
        unet.set_attn_processor(attn_procs)

        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                self._ip_adapter_controlnet_attention_processors = {}
                for controlnet in self.pipe.controlnet.nets:
                    self._default_controlnet_attention_processors[controlnet] = controlnet.attn_processors
                    cn_attn = CNAttnProcessor(num_tokens=self.num_tokens)
                    controlnet.set_attn_processor(cn_attn)
                    self._ip_adapter_controlnet_attention_processors[controlnet] = cn_attn
            else:
                self._default_controlnet_attention_processors = self.pipe.controlnet.attn_processors
                cn_attn = CNAttnProcessor(num_tokens=self.num_tokens)
                self.pipe.controlnet.set_attn_processor(cn_attn)
                self._ip_adapter_controlnet_attention_processors = cn_attn
        
    def load_ip_adapter(self):
        print("loading ip_adapter...")
        state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])
        
    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds
    
    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale
        
    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=-1,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)
        
        if isinstance(pil_image, Image.Image):
            num_prompts = 1
        else:
            num_prompts = len(pil_image)
        
        if prompt is None:
            prompt = self.pos_prompt
        if negative_prompt is None:
            negative_prompt = self.neg_prompt
            
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds = self.pipe._encode_prompt(
                prompt, device=self.device, num_images_per_prompt=num_samples, do_classifier_free_guidance=True, negative_prompt=negative_prompt)
            negative_prompt_embeds_, prompt_embeds_ = prompt_embeds.chunk(2)
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)
            
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images
        
        return images
    
class IPAdapterXL(IPAdapter):
    """SDXL"""

    def create_embeds(self, 
        pil_image,
        prompt=None,
        negative_prompt=None,
        num_samples=1,
        scale=1.0):

        pil_image = prep_image(pil_image)

        self.set_scale(scale)

        if isinstance(pil_image, Image.Image):
            num_prompts = 1
        else:
            num_prompts = len(pil_image)
        
        if prompt is None:
            prompt = self.pos_prompt
        if negative_prompt is None:
            negative_prompt = self.neg_prompt
            
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.pipe.encode_prompt(
                prompt, num_images_per_prompt=num_samples, do_classifier_free_guidance=True, negative_prompt=negative_prompt)
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        embeds = prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

        return embeds

    
    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=1,
        seed=-1,
        num_inference_steps=30,
        **kwargs,
    ):
        
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.create_embeds(
            pil_image, prompt=prompt, negative_prompt=negative_prompt, scale=scale, num_samples=num_samples)
            
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images
        
        return images
    
    
class IPAdapterPlus(IPAdapter):
    """IP-Adapter with fine-grained features"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4
        ).to(self.device, dtype=torch.float16)
        return image_proj_model
    
    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(torch.zeros_like(clip_image), output_hidden_states=True).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds


class IPAdapterPlusXL(IPAdapter):
    """SDXL"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4
        ).to(self.device, dtype=torch.float16)
        return image_proj_model
    
    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(torch.zeros_like(clip_image), output_hidden_states=True).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds
    
    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=-1,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)
        
        if isinstance(pil_image, Image.Image):
            num_prompts = 1
        else:
            num_prompts = len(pil_image)
        
        if prompt is None:
            prompt = self.pos_prompt
        if negative_prompt is None:
            negative_prompt = self.neg_prompt
            
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.pipe.encode_prompt(
                prompt, num_images_per_prompt=num_samples, do_classifier_free_guidance=True, negative_prompt=negative_prompt)
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)
            
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images
        
        return images
