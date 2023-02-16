import os
import requests
import base64
import time
import json
import random
import numpy as np
import cv2
import hashlib
from io import BytesIO
import PIL
from PIL import Image
import torch
from einops import rearrange, repeat
from skimage.exposure import match_histograms

def patch_conv(**patch):
    # Enables tileable textures
    # https://github.com/TomMoore515/material_stable_diffusion
    # Simply call patch_conv(padding_mode='circular') before creating the model
    cls = torch.nn.Conv2d
    init = cls.__init__
    def __init__(self, *args, **kwargs):
        return init(self, *args, **kwargs, **patch)
    cls.__init__ = __init__

def print_model_info(pipe):
    # Find all the parameters in all of the underlying nn modules and count them up:
    module_names = ["vae", "text_encoder", "unet"]
    modules = [pipe.vae, pipe.text_encoder, pipe.unet]
    total_n_params = 0
    for i, m in enumerate(modules):
        if m is not None:
            num_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f"{num_params/1000000.:.2f}M params in {module_names[i]}")
            total_n_params += num_params
    print(f"Total number of parameters: {total_n_params/1000000.:.2f}M")

class DataTracker():
    'Convenience class to save custom tracking numerical data to disk'
    def __init__(self, keys = None):
        self.dataset_path = "/home/xander/Projects/cog/stable-diffusion-dev/eden/xander/splitting_dataset.npz"
        self.data = {}

        self.is_active = False # This class is currently disabled

        if keys is not None:
            for key in keys:
                self.data[key] = []

    def add(self, data_dict):
        if not self.is_active:
            return

        # check if all the keys are already in the data_dict:
        for key in data_dict.keys():
            if key not in self.data.keys():
                self.data[key] = []

        # add the sample data:
        for key in data_dict.keys():
            self.data[key].append(data_dict[key])

    def print_info(self):
        for key in self.data.keys():
            print(key, len(self.data[key]))

    def save(self):
        if not self.is_active:
            return

        # first, create a copy of the current dataset file:
        if os.path.exists(self.dataset_path):
            backup_path = self.dataset_path.replace(".npz", "_backup.npz")
            os.system("cp %s %s" %(self.dataset_path, backup_path))

        # convert all lists to numpy arrays:
        for key in self.data.keys():
            self.data[key] = np.array(self.data[key])

        # load the existing data:
        if os.path.exists(self.dataset_path):
            existing_data = np.load(self.dataset_path, allow_pickle = True)
            for key in existing_data.keys():
                self.data[key] = np.concatenate([existing_data[key], self.data[key]])

        print("Updating dataset at %s!!" %self.dataset_path)
        print("---> Now contains %d samples!" %len(self.data[key]))

        # save the data_dict to disk as compressed numpy array:
        np.savez_compressed(self.dataset_path, **self.data)

        # clear the data_dict:
        self.data = {}

class WaterMarker():
    'Convenience class to add a transparent watermark to video frames'
    def __init__(self, frame_w, frame_h, watermark_path, 
                watermark_size = 0.10,   # as area fraction of full frame area
                watermark_alpha = 0.6,  
                watermark_location = "bottom_right", # (center, bottom_right)
                margin = 0.03            # as fraction of full frame w/h (only applies to bottom_right)
                ):

        print("Creating watermarker!")
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.margin  = margin
        self.watermark_location = watermark_location
        self.watermark_size = watermark_size
        self.watermark_alpha = watermark_alpha
        self.prepare_watermark(watermark_path)

    def prepare_watermark(self, img_data):
        # Load watermark png image, maintaining alpha channel:
        watermark = load_img(img_data, "RGBA")

        # get watermark size:
        watermark_w, watermark_h = watermark.size

        # resize the watermark so its area is equal to self.watermark_size * frame area:
        frame_area = self.frame_w * self.frame_h

        # Solve these equations for new_w and new_h:
        # new_w * new_h = self.watermark_size * frame_area
        # new_w / new_h = watermark_w / watermark_h
        new_w = int(np.sqrt(self.watermark_size * frame_area * watermark_w / watermark_h))
        new_h = int(np.sqrt(self.watermark_size * frame_area * watermark_h / watermark_w))

        # resize watermark:
        self.watermark = watermark.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

        # multiply alpha channel with self.watermark_alpha:
        watermark_np = np.array(self.watermark)
        watermark_np[:, :, 3] = watermark_np[:, :, 3] * self.watermark_alpha
        self.watermark = Image.fromarray(watermark_np)

        # compute pixel margins:
        self.margin_w = int(self.frame_w * self.margin)
        self.margin_h = int(self.frame_h * self.margin)

        # Compute the location in the frame where the watermark will be placed:
        if self.watermark_location == "center":
            self.x = int((self.frame_w - self.watermark.size[0]) / 2)
            self.y = int((self.frame_h - self.watermark.size[1]) / 2)
        elif self.watermark_location == "bottom_right":
            self.x = self.frame_w - self.watermark.size[0] - self.margin_w
            self.y = self.frame_h - self.watermark.size[1] - self.margin_h
        else:
            raise Exception("Unknown watermark location")

    @torch.no_grad()
    def apply_watermark(self, frame):
        for i in range(len(frame)):
            frame[i] = self.add_watermark_to_pil_image(frame[i])
        return frame

    def add_watermark_to_pil_image(self, pil_imgage):
        pil_imgage = pil_imgage.convert("RGBA")
        pil_imgage.paste(self.watermark, (self.x, self.y), self.watermark)
        pil_imgage = pil_imgage.convert("RGB")
        return pil_imgage


class Timer():
    'convenience function to time code'
    def __init__(self, name = ''):
        self.name = name
        self.start = time.time()
    def exit(self, *args):
        print(f'{self.name} took {time.time() - self.start:.3f} seconds')

def get_prompts_from_json_dir(json_dir, shuffle = False):
    json_files = [f for f in sorted(os.listdir(json_dir)) if f.endswith('.json')]
    if shuffle:
        random.shuffle(json_files)
        
    text_inputs = []
    for json_file in json_files:
        with open(os.path.join(json_dir, json_file)) as json_file:
            try:
                prompt_from_disk = json.load(json_file)['text_input']
                text_inputs.append(prompt_from_disk)
            except:
                continue

    return text_inputs
    

def seed_everything(seed):
    if seed is not None:
        seed = int(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)




def preprocess_image(image, shape):
    image = image.resize(shape, resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def preprocess_mask(mask_image, shape):
    mask_w_h = (shape[-1], shape[-2])
    mask = mask_image.resize(mask_w_h, resample=Image.Resampling.LANCZOS)
    mask = mask.convert("L")
    return mask

def sample_from_cv2(sample: np.ndarray) -> torch.Tensor:
    sample = ((sample.astype(float) / 255.0) * 2) - 1
    sample = sample[None].transpose(0, 3, 1, 2).astype(np.float16)
    sample = torch.from_numpy(sample)
    return sample

def sample_to_cv2(sample: torch.Tensor, type=np.uint8) -> np.ndarray:
    sample_f32 = rearrange(sample.squeeze().cpu().numpy(), "c h w -> h w c").astype(np.float32)
    sample_f32 = ((sample_f32 * 0.5) + 0.5).clip(0, 1)
    sample_int8 = (sample_f32 * 255)
    return sample_int8.astype(type)

def sample_from_pil(pil_img): # not tested!!!
    pixel_values = np.array(pil_img)
    sample = sample_from_cv2(pixel_values)
    return sample

def sample_to_pil(sample):
    """Converts a pytorch tensor in the range [0,1] to a PIL image in range [0,255]"""

    sample_f32 = rearrange(sample.squeeze().cpu().numpy(), "c h w -> h w c").astype(np.float32)
    sample_int8 = (sample_f32 * 255).astype(np.uint8)
    return Image.fromarray(sample_int8)


def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample=PIL.Image.Resampling.LANCZOS))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image

def pil_img_to_latent(img, args, device, pipe):
    img = preprocess(img)
    img = img.to(device=device, dtype=pipe.vae.dtype)
    latent = pipe.vae.encode(img).latent_dist.sample(pipe.generator)
    latent = pipe.vae.config.scaling_factor * latent
    return latent

def load_img(data, mode):
    if data.startswith('http://') or data.startswith('https://'):
        image = Image.open(requests.get(data, stream=True).raw)
    elif data.startswith('data:image/'):
        image = load_base64(data, mode)
    elif isinstance(data, str):
        assert os.path.exists(data), f'File {data} does not exist'
        image = Image.open(data)
    image = image.convert(mode)
    return image





def round_to_nearest_multiple(number, multiple):
    return int(multiple * round(number / multiple))
    
def create_seeded_noise(seed, args, device, batch_size=1):
    seed_everything(seed)
    shape = [args.C, args.H // args.f, args.W // args.f]
    random_noise = torch.randn([batch_size, *shape], device=device)
    return random_noise
    
def match_aspect_ratio(n_pixels, img):
    aspect_ratio = np.array(img).shape[1] / np.array(img).shape[0]
    w2 = round_to_nearest_multiple(np.sqrt(n_pixels * aspect_ratio), 64)
    h2 = round_to_nearest_multiple(np.sqrt(n_pixels / aspect_ratio), 64)
    return w2, h2


def load_base64(data, mode):
    data = data.replace('data:image/png;base64,', '')
    pil_img = PIL.Image.open(BytesIO(base64.b64decode(data)))
    pil_img = pil_img.convert(mode)
    return pil_img

def prepare_mask(mask_image, mask_shape, mask_brightness_adjust=1.0, mask_contrast_adjust=1.0, invert_mask=False):
    # PIL mask image
    # shape (list-like len(4)): shape of the image to match, usually latent_image.shape
    # mask_brightness_adjust (non-negative float): amount to adjust brightness of the iamge, 
    #     0 is black, 1 is no adjustment, >1 is brighter
    # mask_contrast_adjust (non-negative float): amount to adjust contrast of the image, 
    #     0 is a flat grey image, 1 is no adjustment, >1 is more contrast
                            
    mask = preprocess_mask(mask_image, mask_shape)

    # Mask brightness/contrast adjustments
    if mask_brightness_adjust != 1:
        mask = TF.adjust_brightness(mask, mask_brightness_adjust)
    if mask_contrast_adjust != 1:
        mask = TF.adjust_contrast(mask, mask_contrast_adjust)

    # Mask image to array
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask,(4,1,1))
    mask = np.expand_dims(mask,axis=0)
    mask = torch.from_numpy(mask)

    if invert_mask:
        mask = ( (mask - 0.5) * -1) + 0.5
    
    mask = np.clip(mask,0,1)
    return mask


def maintain_colors(prev_img, color_match_sample, mode):
    if mode == 'Match Frame 0 RGB':
        return match_histograms(prev_img, color_match_sample, multichannel=True)
    elif mode == 'Match Frame 0 HSV':
        prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
        color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
        matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, multichannel=True)
        return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
    else: # Match Frame 0 LAB
        prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
        color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
        matched_lab = match_histograms(prev_img_lab, color_match_lab, multichannel=True)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)


def add_noise(sample: torch.Tensor, noise_amt: float):
    return sample + torch.randn(sample.shape, device=sample.device) * noise_amt


def get_file_sha256(filepath): 
    sha256_hash = hashlib.sha256()
    with open(filepath,"rb") as f:
        for byte_block in iter(lambda: f.read(4096),b""):
            sha256_hash.update(byte_block)
        sha = sha256_hash.hexdigest()
    return sha


def get_centre_crop(img, aspect_ratio):
    h, w = np.array(img).shape[:2]
    if w/h > aspect_ratio:
        # crop width:
        new_w = int(h * aspect_ratio)
        left = (w - new_w) // 2
        right = (w + new_w) // 2
        crop = img[:, left:right]
    else:
        # crop height:
        new_h = int(w / aspect_ratio)
        top = (h - new_h) // 2
        bottom = (h + new_h) // 2
        crop = img[top:bottom, :]
    return crop


def get_initial_sigma(model_wrap, init_image_strength):
    # When interpolating with very small t_diff steps, we need to make sure that the starting points have different noise
    # To make sure this is always the case, do a very precise calculation of the init noise level (instead of picking the closest discrete point)
    # Perform linear interpolation using the actual float value of args.init_image_strength (instead of rounding to the nearest match)
    start_index = int((init_image_strength) * 1000)
    return model_wrap.get_sigmas(1000)[start_index]

def get_k_sigmas(model_wrap, init_image_strength, steps):
    # Compute the number of remaining denoising steps:
    t_enc = int((1.0-init_image_strength) * steps)

    # Noise schedule for the k-diffusion samplers:
    k_sigmas = model_wrap.get_sigmas(steps)

    # Extract the final sigma-noise levels to use for denoising:
    k_sigmas = k_sigmas[len(k_sigmas)-t_enc-1:]

    #k_sigmas[0] = get_initial_sigma(model_wrap, init_image_strength)

    return k_sigmas, t_enc
    

def lerp(t, v0, v1):
    '''
    Linear interpolation
    Args:
        t (float/np.ndarray): Float value between 0.0 and 1.0
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
    Returns:
        v2 (np.ndarray): Interpolation vector between v0 and v1
    '''
    inputs_are_torch = False
    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    v2 = (1 - t) * v0 + t * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


def slerp(t, v0, v1, flatten = 0, normalize = 0, DOT_THRESHOLD=0.9995, long_arc = 0):
    '''
    Spherical linear interpolation
    Assumes the last dimension of v0 and v1 contains the actual data (this dimension will be auto-normalized)
    Args:
        t (float/np.ndarray): Float value between 0.0 and 1.0
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as
                            colineal. Not recommended to alter this.
        long_arc: if True, interpolates along the long arc of the unit sphere (longest path instead of shortest path)
    Returns:
        v2 (np.ndarray): Interpolation vector between v0 and v1
    '''

    inputs_are_torch = False
    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    # If the vectors are too similar, slerping does weird things, so we just lerp:
    v0_flat = v0.flatten()
    v1_flat = v1.flatten()
    v0_flat_normalized = v0_flat / np.linalg.norm(v0_flat)
    v1_flat_normalized = v1_flat / np.linalg.norm(v1_flat)

    if np.dot(v0_flat_normalized, v1_flat_normalized) > DOT_THRESHOLD:
        print("Input vectors for slerp are too similar, using lerp instead of slerp")
        if inputs_are_torch:
            v0 = torch.from_numpy(v0).to(input_device)
            v1 = torch.from_numpy(v1).to(input_device)
        return lerp(t, v0, v1)
        
    if flatten:
        input_shape = v0.shape
        v0 = v0.flatten()
        v1 = v1.flatten()

    if normalize:
        v0_norms = np.linalg.norm(v0)
        v1_norms = np.linalg.norm(v1)
        normalization_constant = (1-t)* v0_norms + t * v1_norms
        v0 /= v0_norms
        v1 /= v1_norms

    dot = np.sum(v0 * v1) # this is equivalent to np.dot(v0.flatten(), v1.flatten())
    theta_0 = np.arccos(dot)

    if long_arc: # Go around the sphere via the longest path:
        theta_0 = 2*np.pi - theta_0

    # https://en.wikipedia.org/wiki/Slerp
    s0 = np.sin((1 - t)*theta_0) / np.sin(theta_0)
    s1 = np.sin(theta_0 * t) / np.sin(theta_0)
    s0 = np.nan_to_num(s0, nan=0.5)
    s1 = np.nan_to_num(s1, nan=0.5)
    v2 = s0 * v0 + s1 * v1

    if normalize:
        v2 = v2 / np.linalg.norm(v2)
        v2 *= normalization_constant

    if flatten:
        v2 = v2.reshape(input_shape)

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


def get_uniformly_sized_crops(imgs, target_n_pixels):
    """
    Given a list of images:
        - extract the best possible centre crop of same aspect ratio for all images
        - rescale these crops to have ~target_n_pixels
        - return resized images
    """

    # Load images
    #assert len(imgs) > 1
    imgs = [load_img(img_data, 'RGB') for img_data in imgs]
    assert all(imgs), 'Some images were not loaded successfully'
    imgs = [np.array(img) for img in imgs]
    
    # Get center crops at same aspect ratio
    aspect_ratios = [img.shape[1] / img.shape[0] for img in imgs]
    final_aspect_ratio = np.mean(aspect_ratios)
    crops = [get_centre_crop(img, final_aspect_ratio) for img in imgs]

    # Compute final w,h using final_aspect_ratio and target_n_pixels:
    final_h = np.sqrt(target_n_pixels / final_aspect_ratio)
    final_w = final_h * final_aspect_ratio
    final_h = round_to_nearest_multiple(final_h, 64)
    final_w = round_to_nearest_multiple(final_w, 64)

    # Resize images
    resized_imgs = [cv2.resize(crop, (final_w, final_h), cv2.INTER_CUBIC) for crop in crops]
    resized_imgs = [Image.fromarray(img) for img in resized_imgs]
    
    return resized_imgs

import moviepy.editor as mpy
def write_video(frames_dir, video_filename, loop=False, fps=30, codec = 'libx264'):
    frames = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    if loop:
        frames += frames[::-1]
    clip = mpy.ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(video_filename, codec = codec, ffmpeg_params = ["-crf", "20"])


def add_audio_to_video(audio_path, input_video_path, output_video_path, remove_orig_video = True):
    """
    ffmpeg -i input_video.mp4 -i music.mp3 -map 0:v -map 1:a -c:v copy -shortest final.mp4
    """
    
    cmd = f"ffmpeg -i {input_video_path} -i {audio_path} -map 0:v -map 1:a -c:v copy -shortest {output_video_path}"
    os.system(cmd)

    if remove_orig_video:
        os.remove(input_video_path)


def save_settings(args, settings_filename):
    with open(settings_filename, 'w') as f:
        json.dump(vars(args), f, default=lambda o: '<not serializable>', indent=2)



################### prompt utils ###################    



def chunk_prompts(text_inputs, word_range = [2,4], versions = 4):
    all_chunks = []

    for i in range(versions):
        for prompt in text_inputs:
            words = prompt.split(' ')
            chunk, chunk_l = '', 0
            max_n_chunks = random.choice(range(word_range[0], word_range[1], 1))

            for word in words:
                chunk = chunk + ' ' + word
                chunk_l += 1
                if chunk_l >= max_n_chunks:
                    all_chunks.append(chunk)
                    chunk, chunk_l = '', 0
                    max_n_chunks = random.choice(range(word_range[0], word_range[1], 1))

    return all_chunks

def remove_repeated_words(sentence):
    filtered_sentence, prev_word = '', ''
    words = sentence.split(' ')
    for word in words:
        if word != prev_word:
            filtered_sentence = filtered_sentence + ' ' + word
        prev_word = word

    return filtered_sentence

def create_prompts_from_chunks(chunks, chunk_range = [5,20], n = 10000):
    prompts = []
    for i in range(n):
        n_chunks = random.choice(range(chunk_range[0], chunk_range[1], 1))
        sampled_chunks = random.sample(chunks, n_chunks)

        prompt = ''
        for c in sampled_chunks:
            prompt = prompt + " " + c
            if len(prompt) > 180:
                break

        prompt = prompt.replace("   ", " ")
        prompt = prompt.replace("  ", " ")
        prompt = remove_repeated_words(prompt)
        prompt = prompt.strip(" ,;-_.")

        prompts.append(prompt)

    return prompts

def get_cut_ups_from_json_dir(json_dir):
    json_files = [f for f in sorted(os.listdir(json_dir)) if f.endswith('.json')]
    random.shuffle(json_files)
    text_inputs = []
    for json_file in json_files:
        with open(os.path.join(json_dir, json_file)) as json_file:
            try:
                prompt_from_disk = json.load(json_file)['text_input']
                text_inputs.append(prompt_from_disk)
            except:
                continue
    
    chunks = chunk_prompts(text_inputs)
    return create_prompts_from_chunks(chunks)

if __name__ == '__main__':
    pick_best_gpu_id()