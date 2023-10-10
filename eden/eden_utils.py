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
from PIL import Image, ExifTags
import torch
from einops import rearrange, repeat
import moviepy.editor as mpy
import subprocess
import signal

def pick_best_gpu_id():
    # pick the GPU with the most free memory:
    gpu_ids = [i for i in range(torch.cuda.device_count())]
    print(f"# of visible GPUs: {len(gpu_ids)}")
    gpu_mem = []
    for gpu_id in gpu_ids:
        free_memory, tot_mem = torch.cuda.mem_get_info(device=gpu_id)
        gpu_mem.append(free_memory)
        print("GPU %d: %d MB free" %(gpu_id, free_memory / 1024 / 1024))    
    best_gpu_id = gpu_ids[np.argmax(gpu_mem)]
    # set this to be the active GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_id)
    print("Using GPU %d" %best_gpu_id)
    return best_gpu_id

def print_gpu_info(args, status_string, delimiter_string = "-", line_length = 80):
    if args.gpu_info_verbose is False:
        return
    
    timestamp = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    margin_l = (line_length - len(status_string)-2)//2
    print(f"{delimiter_string*margin_l} {status_string} {delimiter_string*margin_l}")
    print(f"--- Time: {timestamp}")
    gpu_ids = [i for i in range(torch.cuda.device_count())]
    for gpu_id in gpu_ids:
        free_memory, tot_mem = torch.cuda.mem_get_info(device=gpu_id)
        print(f"--- GPU {gpu_id}: {(free_memory / 1024 / 1024):.0f} MB available (of {(tot_mem / 1024 / 1024):.0f} total MB)")    
    print(f"{delimiter_string*line_length}")

def patch_conv(**patch):
    # Enables tileable textures
    # https://github.com/TomMoore515/material_stable_diffusion
    # Simply call patch_conv(padding_mode='circular') before creating the model
    cls = torch.nn.Conv2d
    init = cls.__init__
    def __init__(self, *args, **kwargs):
        return init(self, *args, **kwargs, **patch)
    cls.__init__ = __init__


import matplotlib.pyplot as plt
from collections import defaultdict

def inspect_module(ax, nn_module, module_name, total_params):
    layer_map = {'att': 'attention', 'conv': 'conv', 'emb': 'embedding', 
                 'norm': 'norm', 'fc': 'fc', 'proj': 'fc'}
    
    layer_color_map = {'attention': '#3498db', 'conv': '#e67e22', 'embedding': '#2ecc71', 
                    'norm': '#e74c3c', 'fc': '#34495e', 'unk': '#95a5a6'}
    
    n_params, layer_types = [], []
    
    for name, param in nn_module.named_parameters():
        layer_type = 'unk'
        for keyword, ltype in layer_map.items():
            if keyword in name:
                layer_type = ltype
                break
        if param.numel() > 1024:
            n_params.append(param.numel())
            layer_types.append(layer_type)

    cumulative_n_params = 0
    for n_param, layer_type in zip(n_params, layer_types):
        color = layer_color_map[layer_type]
        ax.barh(module_name, n_param, left=cumulative_n_params, color=color, label=layer_type)
        cumulative_n_params += n_param

    # remove the x and y axis completely:
    ax.set_axis_off()

    #ax.set_xlabel('Network layers from left to right (width = n_params)', fontsize=14)
    ax.set_title(f'# params per layer of {module_name} ({total_params/1000000.:.0f}M total params): ', fontsize=20)

    # Define a threshold for the minimum number of total parameters
    min_total_params = total_params/1000

    # Calculate the total number of parameters for each layer type
    total_params = defaultdict(int)
    for n_param, layer_type in zip(n_params, layer_types):
        total_params[layer_type] += n_param

    # Create a consistent legend mapping, excluding layer types with a low number of total parameters
    handles = [plt.Rectangle((0,0),1,1, color=layer_color_map[layer_type]) for layer_type in layer_color_map.keys() if total_params[layer_type] >= min_total_params]
    labels = [layer_type for layer_type in layer_color_map.keys() if total_params[layer_type] >= min_total_params]

    ax.legend(handles, labels, loc='upper right', fontsize=12)

def inspect_total_params(ax, modules, module_names, total_params_list):
    cumulative_n_params = 0
    # Define color map for the modules
    color_map = {'text_encoder': '#1f77b4',  # Blue shade for text_encoder
             'text_encoder_2': '#3498db',  # Different blue shade for text_encoder_2
             'vae': '#8CBF26',  # Matching green
             'unet': '#A35817'}  # Matching brown

    for n_param, module_name in zip(total_params_list, module_names):
        color = color_map.get(module_name, '#d62728')  # default to a color if the module_name isn't found in the color_map
        ax.barh('Total', n_param, left=cumulative_n_params, color=color, label=module_name)
        cumulative_n_params += n_param

    # remove the x and y axis completely:
    ax.set_axis_off()

    ax.set_title(f'Total # params per module ({sum(total_params_list)/1000000.:.0f}M total params): ', fontsize=30)
    ax.legend(loc='upper right', fontsize=12)

def print_model_info(pipe, plot=0):
    module_names = ["text_encoder", "vae", "unet"]
    modules = [pipe.text_encoder, pipe.vae, pipe.unet]

    try:
        modules.insert(1, pipe.text_encoder_2)
        module_names.insert(1, "text_encoder_2")
    except:
        pass

    total_params_list = []
    for m in modules:
        if m is not None:
            total_params_list.append(sum(p.numel() for p in m.parameters()))
        else:
            total_params_list.append(0)

    max_total_params = max(total_params_list)  # Get the maximum number of parameters

    if plot:
        fig, axs = plt.subplots(len(modules) + 1, 1, figsize=(18, 6 * (len(modules) + 1)))
        inspect_total_params(axs[0], modules, module_names, total_params_list)

    for i, m in enumerate(modules):
        total_params = total_params_list[i]
        print(f"{module_names[i]}: {total_params / 1000000.:.2f}M params")

        if plot:
            inspect_module(axs[i + 1], m, module_names[i], total_params)
            # axs[i].set_xlim(0, max_total_params)  # Set a common x-axis limit

    if plot:
        plt.tight_layout()
        plt.savefig("modulevis.png")
        plt.clf()
        plt.close()




def reorder_timepoints(timepoints, verbose = 0):
    """
    given a monotonically increasing array of points, reorder them so that every point divides the largest remaining interval into two equal parts
    This is needed because the iterative smoothing algorithm works this way (LatentBlending based on neighbouring latents)
    """
    timepoints = np.sort(timepoints)
    min_v = int(np.min(timepoints))
    assert min_v == 0, "timepoints must start at 0"
    max_v = int(np.ceil(np.max(timepoints)))

    # first, make sure that all keyframe timepoints are rounded integers:
    integers = np.arange(max_v+1)
    for i in integers:
        # find the closest value in reordered_timepoints and replace it with i:
        closest_index = np.argmin(np.abs(np.array(timepoints) - i))
        timepoints[closest_index] = i

    if verbose:
        print(timepoints)

    reordered_timepoints = []

    for phase in range(max_v):

        # get all the timepoints that are within this phase:
        if len(reordered_timepoints) == 0:
            phase_timepoints = np.array([t for t in timepoints if t >= phase and t <= phase+1])
        else:
            phase_timepoints = np.array([t for t in timepoints if t > phase and t <= phase+1])

        phase_timepoints = np.sort(phase_timepoints)
        phase_indices_to_add = np.arange(len(phase_timepoints))

        reordered_indices_this_phase = []

        # Start by adding the first and last index of this phase:
        if len(reordered_timepoints) == 0:
            reordered_indices_this_phase.append(phase_indices_to_add[0])
            phase_indices_to_add = np.delete(phase_indices_to_add, 0)

        reordered_indices_this_phase.append(phase_indices_to_add[-1])
        phase_indices_to_add = np.delete(phase_indices_to_add, -1)

        # now, interatively add the remaining indices, so that every add index is the one with the largest distance to any of the already added indices:
        while len(phase_indices_to_add) > 0:
            # get the distance between the remaining indices and the already added indices:
            distances = np.abs(phase_indices_to_add[:, None] - np.array(reordered_indices_this_phase)[None, :])
            # get the minimum distance for each remaining index:
            min_distances = np.min(distances, axis = 1)
            # get the index of the remaining index with the largest distance to any of the already added indices:
            max_distance_index = np.argmax(min_distances)

            # add this index to the list of already added indices:
            reordered_indices_this_phase.append(phase_indices_to_add[max_distance_index])

            # remove this index from the remaining indices:
            phase_indices_to_add = np.delete(phase_indices_to_add, max_distance_index)

        # get the actual timepoint values using the reorderforce_timepointsed indices:
        reorder_timepoints_this_phase = phase_timepoints[reordered_indices_this_phase]

        # add the timepoints of this phase to the list of all timepoints:
        reordered_timepoints.extend(reorder_timepoints_this_phase)

    if verbose:
        print("reordered_timepoints:", reordered_timepoints)

    return reordered_timepoints


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

import time
class Timer():
    'convenience class to time code'
    def __init__(self, name, start = False):
        self.name = name
        self.total_time_running = 0.0
        if start:
            self.start()

    def pause(self):
        self.total_time_running += time.time() - self.last_start

    def start(self):
        self.last_start = time.time()

    def status(self):
        print(f'{self.name} accumulated {self.total_time_running:.3f} seconds of runtime')

    def exit(self, *args):
        self.total_time_running += time.time() - self.last_start
        print(f'{self.name} took {self.total_time_running:.3f} seconds')


def get_prompts_from_json_dir(json_dir, shuffle = False, return_seeds = False):
    json_files = [f for f in sorted(os.listdir(json_dir)) if f.endswith('.json')]
    if shuffle:
        random.shuffle(json_files)
        
    text_inputs = []
    seeds = []
    for json_file in json_files:
        with open(os.path.join(json_dir, json_file)) as json_file:
            try:
                data = json.load(json_file)
                text_inputs.append(data['text_input'])
                seeds.append(data['seed'])
            except:
                continue
    if return_seeds:
        return text_inputs, seeds
    else:
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

def pil_img_to_latent(img, args, device, pipe, noise_seed = 0):

    dtype = pipe.vae.dtype
    img = preprocess(img)
    img = img.to(device=device, dtype=dtype)

    if pipe.vae.config.force_upcast:
        img = img.float()
        pipe.vae.to(dtype=torch.float32)

    latent_dist = pipe.vae.encode(img).latent_dist
    latent = latent_dist.sample(torch.Generator(device=device).manual_seed(noise_seed))
    latent = pipe.vae.config.scaling_factor * latent

    if pipe.vae.config.force_upcast:
        pipe.vae.to(dtype=torch.float16)

    return latent

def load_image_with_orientation(path, mode = "RGB"):
    img = Image.open(path)

    try:
        # Get orientation tag (if available)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = img._getexif()
        orientation_value = exif.get(orientation)

        # Apply rotation based on orientation
        if orientation_value == 3:
            img = img.rotate(180, expand=True)
        elif orientation_value == 6:
            img = img.rotate(270, expand=True)
        elif orientation_value == 8:
            img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # No EXIF data or invalid orientation value; do nothing
        pass

    return img.convert(mode)


def load_img(data, mode="RGB"):
    try:
        if data.startswith('http://') or data.startswith('https://'):
            image = load_image_with_orientation(requests.get(data, stream=True).raw)
        elif data.startswith('data:image/'):
            image = load_base64(data, mode)
        elif isinstance(data, str):
            assert os.path.exists(data), f'File {data} does not exist'
            image = load_image_with_orientation(data)
        image = image.convert(mode)
        return image
    except Exception as e:
        print(f"Error loading init_img from {data}")
        print(e)
        raise ValueError(f"Error loading init_img, {str(e)}")

def round_to_nearest_multiple(number, multiple):
    return int(multiple * round(number / multiple))

def create_seeded_noise(seed, args, device, batch_size=1):
    seed_everything(seed)
    shape = [args.C, args.H // args.f, args.W // args.f]
    random_noise = torch.randn([batch_size, *shape], device=device)
    return random_noise

def match_aspect_ratio(n_pixels, img):
    aspect_ratio = np.array(img).shape[1] / np.array(img).shape[0]
    w2 = round_to_nearest_multiple(np.sqrt(n_pixels * aspect_ratio), 8)
    h2 = round_to_nearest_multiple(np.sqrt(n_pixels / aspect_ratio), 8)
    return w2, h2


def load_base64(data, mode):
    data = data.replace('data:image/png;base64,', '')
    pil_img = PIL.Image.open(BytesIO(base64.b64decode(data)))
    pil_img = pil_img.convert(mode)
    return pil_img

def get_random_imgpath_from_dir(dir_path, n=1, extensions = [".jpg", ".png", ".jpeg", ".webp"]):
    # recursively search for images in dir_path:
    img_paths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            for extension in extensions:
                if file.endswith(extension):
                    img_paths.append(os.path.join(root, file))

    # pick n random images:
    img_paths = random.sample(img_paths, n)

    if n==1:
        return img_paths[0]
    else:
        return img_paths


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
        v0 = v0.cpu().numpy().astype(np.float64)
        v1 = v1.cpu().numpy().astype(np.float64)

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
        print("SLERPING along long arc, this should never happen!")
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
    final_h = round_to_nearest_multiple(final_h, 8)
    final_w = round_to_nearest_multiple(final_w, 8)

    # Resize images
    resized_imgs = [cv2.resize(crop, (final_w, final_h), cv2.INTER_CUBIC) for crop in crops]
    resized_imgs = [Image.fromarray(img) for img in resized_imgs]
    
    return resized_imgs


def write_video(frames_dir, video_filename, loop=False, fps=30, codec = 'libx264'):
    try:
        frames = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        print(f"write_video() found {len(frames)} frames, rendering into video...")
    except:
        print(f"Could not find input frames dir {frames_dir}")
        print("Folder structure looks as follows:")
        parent_folder = os.path.dirname(frames_dir)
        try:
            print(f"Parent folder contents ({parent_folder}):")
            print(sorted(os.listdir(parent_folder)))
        except:
            print("Could not find parent folder either")
            return

        try:
            print(f"Input folder contents ({frames_dir}):")
            print(sorted(os.listdir(frames_dir)))
        except:
            print(f"Could not find input folder {frames_dir}")
            return

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


def run_and_kill_cmd(command, pipe_output=True):
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(0.25)

    # Get output from stdout and stderr
    stdout, stderr = p.communicate()    
    # Print the output to stdout in the main process
    if pipe_output:
        if stdout:
            print("cmd, stdout:")
            print(stdout)
        if stderr:
            print("cmd, stderr:")
            print(stderr)

    p.send_signal(signal.SIGTERM) # Sends termination signal
    p.wait()  # Waits for process to terminate

    # Get output from stdout and stderr
    stdout, stderr = p.communicate()

    # If the process hasn't ended yet
    if p.poll() is None:  
        p.kill()  # Forcefully kill the process
        p.wait()  # Wait for the process to terminate

    # Print the output to stdout in the main process
    if pipe_output:
        if stdout:
            print("cmd done, stdout:")
            print(stdout)
        if stderr:
            print("cmd done, stderr:")
            print(stderr)

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


################### for huemin ###################

def huemin_background_gen(outdir,timestring,index):

    color_map = {10:0,20:30,30:30, 50: 40, 60: 80, 70: 80, 80:90, 100:120, 140:150, 165:0}

    def interp_hue(hue, color_map):
        if hue < min(color_map.keys()) or hue > max(color_map.keys()):
            return hue
        sorted_keys = sorted(color_map.keys())
        for i in range(len(sorted_keys)-1):
            if sorted_keys[i] <= hue <= sorted_keys[i+1]:
                interp_val = (hue - sorted_keys[i]) * (color_map[sorted_keys[i+1]] - color_map[sorted_keys[i]]) / (sorted_keys[i+1] - sorted_keys[i]) + color_map[sorted_keys[i]]
                return interp_val

    hue_val = random.randint(0,179)
    #hue_val = interp_hue(hue_in,color_map)
    spread = random.randint(0,1)
    #print(hue_in,hue_val)
    

    def random_hsv():
        # Generate random pastel color in the HSV color space
        hue = (hue_val + random.randint(-spread, spread)) % 180
        saturation = random.randint(80, 255)
        value = random.randint(80, 255)
        pastel = cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2RGB)
        color = pastel[0][0]
        color = (int(color[0]),int(color[1]),int(color[2]))
        return color

    def random_dark_hsv():
        # Generate random pastel color in the HSV color space
        hue = (hue_val + random.randint(-spread, spread)) % 180
        saturation = random.randint(200, 250)
        value = random.randint(0, 0)
        pastel = cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2RGB)
        color = pastel[0][0]
        color = (int(color[0]),int(color[1]),int(color[2]))
        return color

    def add_gradient_background(canvas, start_color, end_color):
        height, width = canvas.shape[:2]
        start_color = np.array(start_color, dtype=np.uint8)
        end_color = np.array(end_color, dtype=np.uint8)
        gradient = np.linspace(start_color, end_color, width).astype(np.uint8)
        canvas[:] = np.repeat(gradient[np.newaxis, :, :], height, axis=0)
        return canvas

    def add_random_gradient_background(canvas):
        start_color = random_dark_hsv()
        end_color = random_dark_hsv()
        canvas = add_gradient_background(canvas, start_color, end_color)
        return canvas

    def create_canvas(height, width):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        return canvas

    def add_noise(canvas, noise_type, noise_param):
        if noise_type == "gaussian":
            # Add Gaussian noise to the canvas
            canvas = canvas + np.random.normal(0, noise_param, canvas.shape)
        elif noise_type == "salt_and_pepper":
            # Add salt and pepper noise to the canvas
            canvas = np.copy(canvas)
            canvas[np.random.randint(0, canvas.shape[0], int(canvas.size * noise_param * 0.004))] = 255
            canvas[np.random.randint(0, canvas.shape[0], int(canvas.size * (1 - noise_param) * 0.004))] = 0
        else:
            raise ValueError("Invalid noise type. Please specify 'gaussian' or 'salt_and_pepper'.")
        return canvas

    def add_stripes(canvas, stripe_width):
        for i in range(0, canvas.shape[1], stripe_width*2):
            cv2.rectangle(canvas, (i, 0), (i + stripe_width, canvas.shape[0]), (255, 255, 255), -1)
        return canvas

    def add_random_stripes(canvas):
        stripe_height = canvas.shape[0]
        stripe_num = random.randint(0,10)
        stripe_width = random.randint(50, 100)
        stripe_spacing = (canvas.shape[1] - (stripe_num*stripe_width)) // (stripe_num+1)
        for i in range(stripe_num):
            x1 = stripe_spacing + i*(stripe_width+stripe_spacing)
            y1 = 0
            x2 = x1 + stripe_width
            y2 = stripe_height
            color = random_hsv() 
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
        return canvas

    def add_random_circles(canvas):
        num_circles = random.randint(10,50)
        for _ in range(num_circles):
            color = random_hsv()
            center = (random.randint(0, canvas.shape[1]), random.randint(0, canvas.shape[0]))
            radius = random.randint(10, 80)
            cv2.circle(canvas, center, radius, color, -1)
        return canvas

    def liquid_distortion(canvas, strength=0.1, scale=0.0):
        strength = random.randint(0,100)/10
        height, width = canvas.shape[:2]
        dx = strength * np.random.randn(height, width)
        dy = strength * np.random.randn(height, width)
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        map1 = (x + dx).astype(np.float32)
        map2 = (y + dy).astype(np.float32)
        canvas = cv2.remap(canvas, map1, map2, cv2.INTER_LINEAR)
        return canvas

    def add_random_blur(canvas):
        kernel_size = random.randint(3, 10)
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma = random.uniform(1, 2)
        canvas = cv2.GaussianBlur(canvas, (kernel_size, kernel_size), sigma)
        return canvas

    def zoom_in(canvas, zoom_percent):
        height, width = canvas.shape[:2]
        zoom_factor = 1 + zoom_percent / 100
        new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, 0, zoom_factor)
        canvas = cv2.warpAffine(canvas, M, (new_width, new_height))
        canvas = cv2.getRectSubPix(canvas, (width, height), center)
        return canvas

    def draw_random_rectangles_outlines(canvas):
        # Get the height and width of the canvas
        height, width = canvas.shape[:2]
        n_rectangles = random.randint(10, 20)

        for i in range(n_rectangles):
            # Generate random rectangle properties
            color = random_hsv()
            size = np.random.randint(50, 200)
            center = (np.random.randint(0, width), np.random.randint(0, height))
            angle = np.random.choice([45, 90])

            # Generate rectangle points
            points = np.array([[center[0]-size, center[1]-size],
                            [center[0]+size, center[1]-size],
                            [center[0]+size, center[1]+size],
                            [center[0]-size, center[1]+size]], dtype=np.float32)

            # Rotate the rectangle
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            points = cv2.transform(points.reshape(1, -1, 2), rotation_matrix).reshape(-1, 2)

            # Draw the rectangle on the canvas
            cv2.polylines(canvas, [points.astype(int)], True, color, 2)

        return canvas

    def draw_random_filled_rectangles(canvas):
        # Get the height and width of the canvas
        height, width = canvas.shape[:2]
        n_rectangles = random.randint(20, 40)
        w_scale = np.random.randint(50, 250)/1000
        h_scale = np.random.randint(50, 250)/1000
        for i in range(n_rectangles):
            # Generate random rectangle properties
            color = random_hsv()
            W = np.random.randint(100, 150)
            H = np.random.randint(25, 50)
            center = (np.random.normal(loc=width*0.5,scale=width*w_scale), np.random.normal(loc=height*0.48,scale=height*h_scale))
            angle = np.random.choice([0, 45, 90])

            # Generate rectangle points
            points = np.array([[center[0]-W, center[1]-H],
                            [center[0]+W, center[1]-H],
                            [center[0]+W, center[1]+H],
                            [center[0]-W, center[1]+H]], dtype=np.float32)

            # Rotate the rectangle
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            points = cv2.transform(points.reshape(1, -1, 2), rotation_matrix).reshape(-1, 2)

            # Draw the filled rectangle on the canvas
            cv2.fillConvexPoly(canvas, points.astype(int), color, 16)

        return canvas

    canvas = create_canvas(1024, 1024)
    #pastel_color = random_hsv()
    #canvas[:] = (0,0,0)
    canvas = add_random_gradient_background(canvas)
    #canvas = add_random_stripes(canvas)
    #canvas = add_random_circles(canvas)
    #canvas = draw_random_rectangles_outlines(canvas)
    canvas = draw_random_filled_rectangles(canvas)
    canvas = liquid_distortion(canvas, strength=5, scale=5)
    canvas = add_random_blur(canvas)
    canvas = zoom_in(canvas, 10)
    canvas = add_noise(canvas, "gaussian", 5)
    gen_path = os.path.join(outdir,f"{timestring}_{index:05}_gen.jpg")
    cv2.imwrite(gen_path, canvas)
    return gen_path


############ ControlNet input processing ############

def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    # gamma correction
    img = img / 255
    img = np.power(img, 2.2)
    img = img * 255
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    return img

@torch.no_grad()
def preprocess_controlnet_init_image(pil_control_image, args):
    if "depth" in args.controlnet_path:
        return preprocess_zoe_depth(pil_control_image)

    if "canny" in args.controlnet_path:
        return preprocess_canny(pil_control_image)

    if "luminance" in args.controlnet_path:
        if args.interpolator is not None: # video
            threshold = False
            target_width_range = [512,512]
        else: # image
            threshold = False
            target_width_range = [264,264]
            #target_width_range = [48,264]
        return preprocess_luminance(pil_control_image, threshold = threshold, target_width_range=target_width_range)

def compute_brightness_map(image):
    """Compute the perceptual brightness map of an image."""
    # Convert the image to the Lab color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    # Extract the L channel (luminance)
    l_channel = lab[:, :, 0]
    return l_channel

def resize_image(image, target_width):
    """Resize the image to the target width while maintaining the aspect ratio."""
    aspect_ratio = image.shape[1] / image.shape[0]
    target_height = int(target_width / aspect_ratio)
    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    return resized_image

def preprocess_luminance(pil_control_image, 
        threshold=False, 
        target_width_range = None,
        ):
    # convert pil image to cv2 image:
    image = np.array(pil_control_image)
    brightness_map = compute_brightness_map(image)

    if target_width_range is not None: # down- and up- size the brightness map (creates a box-blur effect)
        target_width   = np.random.randint(target_width_range[0], target_width_range[1] + 1)
        brightness_map = resize_image(brightness_map, target_width)
        #brightness_map = cv2.resize(brightness_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        brightness_map = cv2.resize(brightness_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    if threshold: # threshold the brightness map
        _, brightness_map = cv2.threshold(brightness_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # convert the brightness map to a PIL image
    brightness_map = Image.fromarray(brightness_map)
    return brightness_map

from PIL import Image, ImageEnhance, ImageFilter
def preprocess_canny(pil_control_image):
    """
    Takes a PIL image, computes the canny edge map
    and returns a canny edge map PIL image to use in the control net
    """

    image = pil_control_image.convert("RGB")

    if 0:
        # Contrast enhancement: Increase the contrast by 1.5x
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)

        # Sharpening: Apply a sharpening filter
        sharpening_kernel = ImageFilter.Kernel((3, 3), [0, -1, 0, -1, 5, -1, 0, -1, 0])
        image = image.filter(sharpening_kernel)

    image = np.array(image)
    image = cv2.Canny(image, 100, 200)[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return  Image.fromarray(image)


from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers.utils import load_image

def prep_depth_map(image):
    """
    - convert the image to a single channel grayscale img
    - rescale the given PIL image to the 0-1 range
    - duplicate the grayscale image to 3 channels
    """
    image = image.convert("L")
    image = np.array(image)

    image = (image - image.min()) / (image.max() - image.min())
    image = np.stack([image, image, image], axis=2)
    
    return Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))


def preprocess_zoe_depth(image):
    width, height = image.size

    torch.cuda.empty_cache()

    depth_estimator   = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    print("Predicting depth...")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth
    
    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image



    # delete the depth model and free the cuda memory:
    del model_zoe_n
    torch.cuda.empty_cache()

    return depth


if __name__ == '__main__':
    pick_best_gpu_id()



