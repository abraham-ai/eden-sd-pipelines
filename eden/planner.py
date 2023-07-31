import numpy as np
import torch
import torch.nn as nn
import os, time, math
from PIL import Image, ImageEnhance

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 15

from audio import create_audio_features
from eden_utils import pil_img_to_latent, slerp, create_seeded_noise, save_settings, sample_to_pil


def subtract_dc_value(signal):
    values, counts = np.unique(signal, return_counts=True)
    ind = np.argmax(counts)
    dc_value = values[ind]
    return signal - dc_value

class Planner():
    """

    This class allows for dynamic updating of the arguments fed to generate(args), based on the audio features 
    to enable audio-reactive video generation with SD

    """
    def __init__(self, audio_load_path, fps, total_frames):

        self.fps = fps
        self.frame_index = 0
        
        self.load_audio_features(audio_load_path)
        #self.modulation_text_c = get_prompt_conditioning("vibrant colors, high contrast, high saturation, masterpiece, trending on Artstation", 'autocast')
        
        # Compute which fraction of the total audio features are being rendered:
        self.total_frames = total_frames
        # Rendered frames:
        total_frame_time = total_frames / fps
        audio_time = self.metadata["duration_seconds"]
        self.audio_fraction = total_frame_time / audio_time
        print("Rendering frames for ", total_frame_time, " seconds of video, which is ", 100*self.audio_fraction, "% of the total audio time.")

        if self.audio_fraction > 1:
            print("Warning: more video frames than audio features, this will lead to errors!")   

    def __len__(self):
        return len(self.frames)

    def load_audio_features(self, audio_load_path):
        self.harmonic_energy, self.final_percus_features, self.metadata, self.audio_path = create_audio_features(audio_load_path)
        self.prep_audio_signals_for_render()

    def get_audio_push_curve(self, n_samples, prompt_index, n_frames_between_two_prompts, max_n_samples = 99999, verbose = 0):
        """
        Get the audio_energy curve (used to push the perceptual change in the video) at this phase in the interpolation
        and resample it to the target n_samples

        min_n_samples: the minimum number of samples before actally returning audio features, otherwise return 1s
        """
        # select the relevant fraction of the audio features (based on the phase_index in the interpolation):
        start_index, end_index = prompt_index*(n_frames_between_two_prompts + 1), (prompt_index+1)*(n_frames_between_two_prompts + 1)
        current_push_segment = self.push_signal[start_index:end_index]

        # Currently the algo creates keyframe --> keyframe phases of equal # frames
        # A constant, smooth video would result from a constant push_signal = 1
        # The total visual change in such a phase is divided in segments to align with the push_signal as optimally as possible
        # Therefore, we want the push signal in each phase to be normalized:
        current_push_segment = current_push_segment #/ np.mean(current_push_segment)
        # This however has the problem that the local variations in video speed depend on the normalization constant used above
        # So for very quiet audio segments, subtle audio variations will result in large visual changes, whereas for other segments
        # those same audio variations wont be noticeable at all.
        # Ideally, we want to normalize the entire audio signal in one go. 
        # However, this is also not ideal because the some segments would result in a super low "target push signal" (e.g. 0.1) that is unattainable

        if n_samples < (0.1 * max_n_samples): # require at least 10% of the max_n_samples to be already rendered before returning the actual audio features
            return np.ones(n_samples), np.ones_like(current_push_segment)


        # resample the push_signal to match the number of samples:
        old_x = np.linspace(0, 1, current_push_segment.shape[0])
        new_x = np.linspace(0, 1, n_samples)

        resampled_current_push_segment = resample_signal(new_x, old_x, current_push_segment)

        if 0:
            from datetime import datetime
            # create a combined plot of all signals:
            plt.figure(figsize=(14,8))
            plt.plot(old_x, current_push_segment, label="full_signal")
            plt.plot(new_x, resampled_current_push_segment, label="downsampled_signal")
            plt.legend()
            plt.ylim(0.0, 2.0)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            plt.savefig(f"resampling_curve_{timestamp}.png")
            plt.clf()

        #resampled_current_push_segment = resampled_current_push_segment / np.mean(resampled_current_push_segment)

        return resampled_current_push_segment, current_push_segment

    def prep_audio_signals_for_render(self):
        # Make sure the base signal is 0 by default:
        self.final_percus_features[0,:] = subtract_dc_value(self.final_percus_features[0,:])

        # Resample the audio features to match the video fps:
        old_x = np.linspace(0, self.metadata["duration_seconds"], int(self.metadata["duration_seconds"] * self.metadata["features_per_second"]))
        new_x = np.linspace(0, self.metadata["duration_seconds"], int(self.metadata["duration_seconds"] * self.fps))

        self.fps_adjusted_harmonic_energy = resample_signal(new_x, old_x, self.harmonic_energy)

        self.fps_adjusted_percus_features = []
        for i in range(self.final_percus_features.shape[0]):
            self.fps_adjusted_percus_features.append(resample_signal(new_x, old_x, self.final_percus_features[i,:]) )

        self.fps_adjusted_percus_features = np.array(self.fps_adjusted_percus_features)

        self.push_signal = self.fps_adjusted_harmonic_energy
        #self.push_signal = self.fps_adjusted_percus_features[-1, :] + 0.025
        self.push_signal = self.push_signal / np.mean(self.push_signal)

        # plot the harmonic_energy curve:
        if 1:
            seconds = 60
            frames = self.fps*seconds
            plt.figure(figsize=(14,8))
            plt.plot(self.push_signal[:frames])
            plt.ylim(0.0, 2.0)
            plt.savefig("fps_adjusted_percus_features.png")
            plt.clf()

    def morph_image(self, image, 
                    frame_index = None,
                    brightness_factor = 0.004,
                    contrast_factor   = 0.4,
                    saturation_factor = 0.5,
                    zoom_factor       = 0.007,
                    noise_factor      = 0.0,
                    ):
        if frame_index is None:
            frame_index = self.frame_index

        # increase the brightness of the init_img:
        enhancer = ImageEnhance.Brightness(image)
        factor = 1 + brightness_factor * self.fps_adjusted_percus_features[2, frame_index]
        image = enhancer.enhance(factor)

        # increase the contrast of the init_img:
        enhancer = ImageEnhance.Contrast(image)
        factor = 1 + contrast_factor * self.fps_adjusted_percus_features[1, frame_index]
        image = enhancer.enhance(factor)

        # increase the saturation of the init_img:
        enhancer = ImageEnhance.Color(image)
        factor = 1 + saturation_factor * self.fps_adjusted_percus_features[1, frame_index]
        image = enhancer.enhance(factor)

        # slightly crop and zoom in on the init_img:
        factor = 1 + zoom_factor * self.fps_adjusted_percus_features[0, frame_index]
        # get the center pixel coordinates:
        x, y = image.size[0]//2, image.size[1]//2
        image = zoom_at(image, x, y, factor)

        # slightly rotate the init_img:
        # rotation_angle = 0.5 * self.fps_adjusted_percus_features[2, self.frame_index]
        # image = image.rotate(rotation_angle)

        # add random pixel noise to the img:
        if noise_factor > 0:
            noise_img = Image.fromarray(np.uint8(np.random.rand(image.size[1], image.size[0], 3) * 255))
            factor = noise_factor * self.fps_adjusted_percus_features[2, frame_index]
            image = Image.blend(image, noise_img, factor)
        
        return image


    def adjust_args(self, args, t_raw, force_timepoints = None, verbose = 0):

        if force_timepoints is not None:
            # find the index of t_raw in all the timepoints:
            sorted_force_timepoints = np.sort(force_timepoints)
            self.frame_index = np.argmin(np.abs(np.array(sorted_force_timepoints) - t_raw))
            print("Forcing audio frame index to: ", self.frame_index)

        # adjust the args according to the audio features:

        if 0:
            # Modulate the guidance_scale with the percussion:
            # for higher init_img_strength, we want to modulate the guidance_scale more since there's less diffusion steps:
            scale_modulation_f = 12 + 10 * args.init_image_strength
            args.guidance_scale = args.guidance_scale + scale_modulation_f * self.fps_adjusted_percus_features[-1, self.frame_index]
            #args.guidance_scale = args.guidance_scale + scale_modulation_f * self.push_signal[self.frame_index]

        if 0:
            # modulate args.c in the direction of self.modulation_text_c
            conditioning_modulation_f = 0.5 * self.fps_adjusted_percus_features[2, self.frame_index]
            #print("Modulating args.c by factor: ", conditioning_modulation_f)
            args.c = (1-conditioning_modulation_f) * args.c + conditioning_modulation_f*self.modulation_text_c

        if 0:
            if args.init_latent is not None:
                # modulate the init_latent with the percussive features:
                args.init_latent = args.init_latent * (1 - 0.5*self.fps_adjusted_percus_features[0, self.frame_index])

        if 0: # mess with the init img:
            args.init_image = self.morph_image(args.init_image)

        #self.frame_index += 1

        return args

    def respace_audio_timepoints(self, x, target_n_points):
        # given timepoints x (irregularly spaced), corresponding to a smooth, visual trajectory,
        # resample the x-axis to correspond to the given visual change target y (which is also irregularly spaced)

        # Importantly, x is expected to be sampled at the same fps as self.harmonic_energy
        x1, y1 = x, np.linspace(0,1, len(x))

        y2 = self.harmonic_energy[:len(x)]
        y2 = np.cumsum(y2)
        y2 = y2 / y2[-1]

        if 1: # Harmonic energy modulation
            # Now, find x2, corresponding to y2 (from the same function as (x1, y1)):
            x2 = resample_signal(y2, y1, x1)
        else:
            # Dont modulate the timepoints using harmonic energy, simply use a linear time increase:
            x2 = resample_signal(np.linspace(0,1,len(y2)), y1, x1)

        # finally, interpolate x2 to the target_n_points:
        t = np.linspace(0, 1, len(x2))
        t_fin = np.linspace(0, 1, target_n_points)
        x_fin = resample_signal(t_fin, t, x2)

        return x_fin


from scipy import interpolate
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator

def resample_signal(new_x, x, y, interpolation_type = "Akima"):
    # Given a signal y sampled at x, resample it to the new_x points
    if interpolation_type == "Akima":
        new_y = Akima1DInterpolator(x, y)(new_x)
    elif interpolation_type == "Pchip":
        new_y = PchipInterpolator(x, y)(new_x)
    elif interpolation_type == "cubic":
        f = interpolate.interp1d(x, y, kind='cubic')
        new_y = f(new_x)
    elif interpolation_type == "linear":
        f = interpolate.interp1d(x, y, kind='linear')
        new_y = f(new_x)

    # When resampling with fixed time intervals, this is prob better:
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html

    return new_y

def zoom_at(img, x, y, zoom):
    w, h = img.size
    zoom2 = zoom * 2
    img = img.crop((x - w / zoom2, y - h / zoom2, 
                    x + w / zoom2, y + h / zoom2))
    return img.resize((w, h), Image.Resampling.LANCZOS)

def respace_timepoints(timepoints, density, n_to_sample, resolution_f = 1000):
    
    """

    Given an array of monotonically increasing points and their corresponding density,
    resample the points (along the same curve) to have a near constant inter-point density.

    """
    if len(density) == len(timepoints)-1:
        # If the density is given for the intervals between timepoints, lets just repeat the final value:
        density = np.append(density, density[-1])

    x = np.array(timepoints)
    y = np.array(density)

    # Essentially, we need to divide the area under the density curve into n equal parts.
    # To do this, we first approximate the total area under the density curve by summing up
    # all the rectangles of which the height is the density at that timepoint and the width is the distance between timepoints:

    avg_delta = np.mean(np.diff(x))
    t_increment = avg_delta / resolution_f  # determines the smallest step in interpolation space

    new_x, current_area = [x[0]], 0
    new_timepoints = np.arange(x[0], np.max(x), t_increment)
    densities = resample_signal(new_timepoints, x, y)

    total_area = np.trapz(densities, new_timepoints)
    area_per_timepoint = total_area/(n_to_sample-1)

    # Now, we can sample the curve at regular intervals of area_per_timepoint:
    for i, local_density in enumerate(densities):

        if len(new_x) >= n_to_sample:
            print("respace_timepoints --> breaking early because len(new_x) >= n_to_sample")
            break

        current_area += local_density * t_increment

        if current_area >= area_per_timepoint:
            new_x.append(new_timepoints[i])
            current_area = 0
    if 0:
        from datetime import datetime
        # create a combined plot of all signals:
        plt.figure(figsize=(14,8))
        plt.plot(x, y, label="input")
        plt.plot(new_x, resample_signal(new_x, x, y), label="output")
        plt.plot(new_timepoints, densities, label = "upsampled_density")
        plt.legend()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.savefig(f"resampling_curve_{timestamp}.png")
        plt.clf()

    return np.array(new_x)







#####################################################################################
#####################################################################################
#####################################################################################


# LatentBlending / FrameBuffer classes:



def blend_inits(init1, init2, t, args, real2real = True, anti_shinethrough_power = 1.5, only_need_init_strength = False):
    '''
    Create a linear blend between the two inits using t
    These inits can be either pil images or pytorch latent vectors
     - args.interpolation_init_images_power (float): power of the init_img_strength curve (how fast init_strength declines at endpoints)
     - args.interpolation_init_images_min_strength (float): minimum strength of the init_img during interpolation
     - args.interpolation_init_images_max_strength (float): maximum strength of the init_img during interpolation

     anti_shinethrough_power (float): power of the init_img_strength curve (how fast init_strength declines towards midpoint),
    higher value means less shine-through

    '''

    left_strength  = 1-t
    right_strength = t
    
    if real2real:
        # subtly adjust the mixture so the other endpoint img does not "shine through" when init_image_strength is relatively high:
        left_strength  = left_strength ** anti_shinethrough_power
        right_strength = right_strength ** anti_shinethrough_power
        mixing_t = right_strength / (left_strength + right_strength)

        # the init_image_strength starts at 1 for t=0, goes to 0 when t=0.5 and then back to 1 for t=1
        init_image_strength = (abs(t-0.5)*2)**args.interpolation_init_images_power

        # adjust init_image_strength to fall in range [min_init_img_strength, max_init_img_strength]
        mins, maxs = args.interpolation_init_images_min_strength, args.interpolation_init_images_max_strength
        init_image_strength = mins + (maxs - mins) * init_image_strength

    else: # normal prompt2prompt lerping
        mixing_t = t
        init_image_strength = 0.0

    if only_need_init_strength:
        return None, init_image_strength
    
    if isinstance(init1, Image.Image):
        blended_init = (left_strength * np.array(init1) + right_strength * np.array(init2)) / (left_strength + right_strength)
        blended_init = Image.fromarray(blended_init.astype(np.uint8))
    else:
        blended_init = slerp(mixing_t, init1, init2, flatten = 1, normalize = 1)

    return blended_init, init_image_strength


def create_init_latent(args, t, interpolation_init_images, keyframe_index, init_noise, device, pipe):

    """
    This function is derived from the latent-blending idea:
    https://twitter.com/j_stelzer/status/1613179427659169792

    Instead of creating an init_image, directly create the init_latent, which is a
    linear combination of surrounding latent_vectors (at the same timepoint in the diffusion process)
    higher skip_f values will lead to smoother video transitions and lower render time, but also have less interesting transitions

    """

    if not ((args.interpolation_init_images and all(args.interpolation_init_images) or len(args.interpolator.latent_tracker.frame_buffer.ts) >= args.n_anchor_imgs)):
        # anchor_frames for lerp: only use the raw init_latent noise from interpolator (using the input seeds)
        pipe.scheduler.set_timesteps(args.steps, device=device)
        init_latent = init_noise * pipe.scheduler.init_noise_sigma
        init_image = None
        init_image_strength = 0.0

        return init_latent, init_image, init_image_strength

    latent_tracker = args.interpolator.latent_tracker
    real2real = False if interpolation_init_images is None else True

    if real2real:
        init_img0, init_img1 = interpolation_init_images[keyframe_index], interpolation_init_images[keyframe_index + 1]
    else: # lerping mode (no real init imgs, so use the generated keyframes)
        init_img0, init_img1 = latent_tracker.frame_buffer.get_current_keyframe_imgs()
        init_img0, init_img1 = sample_to_pil(init_img0), sample_to_pil(init_img1)

    # Project init images into latent space:
    key_latent0 = pil_img_to_latent(init_img0, args, device, pipe)
    key_latent1 = pil_img_to_latent(init_img1, args, device, pipe)

    init_image, init_latent, timestep  = None, None, None
    
    if (len(latent_tracker.t_raws) < args.n_anchor_imgs or (args.latent_blending_skip_f is None)) and 0:
        print("Simply alpha-blending the keyframe latents..")
        # simply alpha-blend the keyframe latents using t:
        init_latent, init_image_strength = blend_inits(key_latent0, key_latent1, t, args, real2real = real2real)

        if init_image_strength == 1: # first or last frame of the interpolation, just return raw image
            init_latent = None
            init_image = init_img0 if t < 0.5 else init_img1
        else: # TODO
            pass

    elif (len(latent_tracker.t_raws) < args.n_anchor_imgs or (args.latent_blending_skip_f is None)) and 1:
        print("Pixel blending...")
        # apply linear blending of keyframe images in pixel space and then encode
        init_image, init_image_strength = blend_inits(init_img0, init_img1, t, args, real2real = real2real)
        init_latent = None

    else: # Apply Latent-Blending trick:
        print("Latent Blending...")
        _, init_image_strength = blend_inits(key_latent0, key_latent1, t, args, real2real = real2real, only_need_init_strength = True)
        
        # apply latent_blending_skip_f to the init_image_strength:
        # this essentially skips-ahead in the denoising process
        # compute latent_blending_skip_f based on the perceptual distance of the current transition (which we'll split in two) 
        # get the perceptual distance at the current transtion that we're going to split:
        current_lpips_distance = latent_tracker.frame_buffer.get_perceptual_distance_at_t(t)

        # We want a gradual increase of skip_f from min_v when current_lpips_distance >= max_d to max_v when current_lpips_distance < min_d
        min_skip_f, max_skip_f = args.latent_blending_skip_f[0], args.latent_blending_skip_f[1]
        # hardcoded lpips_perceptual distances corresponding to min_v and max_v latent_skip_f values:
        max_d, min_d = 0.65, 0.15 # normal renders
        latent_tracker.latent_blending_skip_f = min_skip_f + (max_skip_f - min_skip_f) * (max_d - current_lpips_distance) / (max_d - min_d)
        latent_tracker.latent_blending_skip_f = np.clip(latent_tracker.latent_blending_skip_f, min_skip_f, max_skip_f)
        init_image_strength = init_image_strength * (1.0 - latent_tracker.latent_blending_skip_f) + latent_tracker.latent_blending_skip_f
        
        # grab the nearest neighbouring latents, at the corresponding timepoint in the diffusion process:
        latent_left, latent_right, t_raw_left, t_raw_right = latent_tracker.get_neighbouring_latents(args, adjusted_init_image_strength = init_image_strength)

        # linearly interpolate between the neighbouring latents for create the new init_latent:
        mixing_f    = (t - t_raw_left%1) / (t_raw_right - t_raw_left)
        init_latent = (1-mixing_f) * latent_left + mixing_f * latent_right

        # Correct the std of the combined latent (summing uncorrelated gaussians decreases the std)
        target_std = (1-mixing_f) * latent_left.std() + mixing_f * latent_right.std()
        if init_latent.std() > 0:
            init_latent = target_std * init_latent / init_latent.std()

    return init_latent, init_image, init_image_strength


class LatentTracker():
    """
    Helper class that stores all intermediate diffusion latents and frames to enable Latent-Blending and smoothing
    originally implemented for https://twitter.com/xsteenbrugge/status/1558508866463219712
    further improvements inspired by: https://twitter.com/j_stelzer/status/1613179427659169792
    """
    def __init__(self, args, pipe, smooth):
        self.latents = {}
        self.init_noises = {}
        self.t_raws = []
        self.pipe = pipe
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_blending_skip_f = 0.0
        self.frame_buffer = FrameBuffer(smooth, args)
        self.phase_data = None      
        self.pipe.scheduler.set_timesteps(args.steps, device=self.device)

    def get_n_frames(self):
        return len(self.frame_buffer.frames)

    def reset_buffer(self):
        # This function is triggered at the start of a new interpolation phase (two new keypoints / prompts)
        if self.args.save_phase_data:
            self.save_to_disk()
            self.phase_data = None

        max_t_raw = np.max(self.t_raws)
        last_frame_latents = self.latents[max_t_raw].copy()
        self.latents = {}
        self.init_noises = {}
        self.t_raws = []

        # add the last frame latents to the new buffer:
        self.latents[max_t_raw]   = last_frame_latents
        self.t_raws.append(max_t_raw)
        self.frame_buffer.clear_buffer()

    def create_new_denoising_trajectory(self, args, pipe):
        # create a new denoising trajectory for the given timestep
        self.current_t_raw = args.t_raw
        self.current_init_image_strength  = args.init_image_strength
        #self.noise = create_seeded_noise(args.seed, args, self.device)
        self.t_raws.append(self.current_t_raw)
        self.latents[self.current_t_raw]   = [None]*(args.steps + 1)
        pipe.scheduler.set_timesteps(args.steps, device=self.device)

    def add_frame(self, args, img_t, t, t_raw):
        self.frame_buffer.add_frame(img_t, t)
        self.add_to_phase_data(args, t_raw)

    def t_to_index(self, t, pre_timestep):
        # pre_timestep is 0 normally,
        # but 1 when the callback is called before the first denoising step (just once per frame)
        return (self.pipe.scheduler.timesteps == t).nonzero(as_tuple=True)[0] + 1 - pre_timestep

    def add_latent(self, i, t, latent, pre_timestep = 0):
        latent = latent.detach().cpu().numpy().astype(np.float32)
        self.latents[self.current_t_raw][self.t_to_index(t, pre_timestep)] = latent

    def get_neighbouring_ts(self, t_raw):
        sorted_t_raws = sorted(self.t_raws)
        # Find the closest t that's smaller than args.t_raw:
        t_raw_left = sorted_t_raws[np.searchsorted(sorted_t_raws, [t_raw], side="left")[0]-1]
        # Find the closest t that's larger than args.t_raw:
        t_raw_right = sorted_t_raws[np.searchsorted(sorted_t_raws, [t_raw], side="right")[0]]
        return t_raw_left, t_raw_right

    def adjust_n_steps(self, new_n_steps):
        # TODO
        # Grab all the final latents for each t
        # Re-encode all the noise stacks
        return

    def add_noise_to_latent(self, torch_fully_denoised_latent, timestep, noise = None):
        if noise is None: # create fully random noise
            generator = torch.Generator(device=self.device).manual_seed(int(time.time()))
            noise = torch.randn(*torch_fully_denoised_latent.shape, generator=generator, device=self.device, dtype=torch_fully_denoised_latent.dtype)
        
        noised_latent = self.pipe.scheduler.add_noise(torch_fully_denoised_latent, noise, timestep.unsqueeze(0)).cpu().numpy()

        return noised_latent

    def construct_noised_latents(self, args, t_raw):
        if 1:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print(f"Constructing noisy stack of latents at t={t_raw:.3f}...")
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # given the current denoising stack of latents for this frame,
        # construct the full stack all the way up to pure noise
        
        fully_denoised_latent = self.latents[t_raw][-1]
        
        try:
            noise_sigmas = self.pipe.scheduler.sigmas
        except:
            noise_sigmas = None

        if fully_denoised_latent is None:
            print("---------- WARNING ----------")
            print("The last (denoised) latent in the stack is None, this should never happen!")

        if self.args.easy_way or 1: #easy way

            torch_fully_denoised_latent = torch.from_numpy(fully_denoised_latent).to(self.device).float()
            # Loop over all the timesteps and add the corresponding noise to the fully_denoised_latent:
            for i in range(len(self.latents[t_raw])):
                if self.latents[t_raw][i] is not None: # we've reached the existing latents in the stack, stop constructing
                    break

                # add_noise() is normally used at the start of the denoising trajectory
                timestep = self.pipe.scheduler.timesteps[i]
                #latents = self.add_noise_to_latent(torch_fully_denoised_latent, timestep, noise = self.init_noises[t_raw])
                latents = self.pipe.scheduler.add_noise(torch_fully_denoised_latent, self.init_noises[t_raw], timestep.unsqueeze(0)).cpu().numpy()

                # Slight hack to make sure we're at the right noise level:
                #latents = latents / np.std(latents)
                #latents = latents * (noise_sigmas[i].cpu().item() + np.std(fully_denoised_latent))
                self.latents[t_raw][i] = latents

        else: # harder way, but might be smoother:
            print("HARD")
            # start at the most noisy latent in the stack, grab the corresponding noise level at that index
            # loop over all the missing indices and add the corresponding amount of fixed_noise
            most_noisy_index  = np.min([i for i, latent in enumerate(self.latents[t_raw]) if latent is not None])
            most_noisy_latent = self.latents[t_raw][most_noisy_index]

            # Grab the expected sigma levels of noise at each level in the stack:
            # we get rid of the first one since the stack starts saving latents after the first denoising step
            
            print("target sigmas:")
            print(np.round(noise_sigmas.cpu().numpy(),3))

            def get_k_sigmas(pipe, init_image_strength, steps):
                pipe.scheduler.set_timesteps(steps, device="cuda")
                # Compute the number of remaining denoising steps:
                t_enc = int((1.0-init_image_strength) * steps)

                # Noise schedule for the k-diffusion samplers:
                k_sigmas_full = pipe.scheduler.sigmas

                # Extract the final sigma-noise levels to use for denoising:
                k_sigmas = k_sigmas_full[len(k_sigmas_full)-t_enc-1:]

                return k_sigmas, k_sigmas_full

            current_sigmas, k_sigmas_full = get_k_sigmas(self.pipe, self.current_init_image_strength, args.steps)
            active_sigmas = k_sigmas_full - current_sigmas[0]
            
            for i in range(len(self.latents[t_raw])):
                if self.latents[t_raw][i] is not None: # we've reached the existing latents in the stack, stop constructing
                    break

                # works for lerp, not for real2real:
                offset1 = 0
                offset2 = 0

                # works for real2real, not for lerp:
                #offset1 = 0
                #offset2 = -1

                noise = self.init_noises[t_raw].cpu().numpy()
                #noise = np.random.normal(size=noise.shape) # create fully random noise

                target_sigma     = noise_sigmas[i + offset1].cpu().item()
                most_noisy_sigma = noise_sigmas[most_noisy_index + offset2].cpu().item()
                sigma_to_add     = target_sigma - most_noisy_sigma

                #print(f"bugged sigma: {sigma_to_add:.3f}, old sigma: {active_sigmas[i].cpu().item():.3f}")
                #sigma_to_add = active_sigmas[i].cpu().numpy()

                latents = most_noisy_latent + sigma_to_add * noise

                # Slight hack to make sure we're at the right noise level:
                #latents = latents / np.std(latents)
                #latents = latents * (noise_sigmas[i].cpu().item() + np.std(fully_denoised_latent))
                #latents = latents * noise_sigmas[i].cpu().item()
                self.latents[t_raw][i] = latents

                if i == 0:
                    print(f"Target sigma: {target_sigma:.3f}")
                    print(f"Most noisy sigma: {most_noisy_sigma:.3f}")
                    print(f"final sigma: {np.std(latents):.3f}")
                
    def get_neighbouring_latents(self, args, adjusted_init_image_strength = None):
        
        if adjusted_init_image_strength is not None:
            self.current_init_image_strength = adjusted_init_image_strength

        # We need to grab a latent in the denoising trajectory that corresponds to the 
        # init_strength level we're currently targetting.
        # Compute the corresponding latent index into the denoising trajectory:

        # This is the cutoff value that will be used in the sampler:
        total_n_steps   = self.pipe.scheduler.config.num_train_timesteps
        timestep_cutoff = int(round(total_n_steps - (self.current_init_image_strength * total_n_steps)))

        # grab the index in the denoising trajectory that corresponds to the previous timestep:
        # the latents at this level will be used to generate the starting latents for the current frame
        for i,t in enumerate(self.pipe.scheduler.timesteps.cpu().numpy()):
            if t < timestep_cutoff:
                target_trajectory_index = i
                break

        # Find the closest t's left and right of the target t_raw:
        t_raw_left, t_raw_right = self.get_neighbouring_ts(args.t_raw)

        # if available, get latents from the neighbouring frames denoising stack at the current index
        latents_left  = self.latents[t_raw_left][target_trajectory_index]
        latents_right = self.latents[t_raw_right][target_trajectory_index]

        print_info = False
        if (latents_left is None) and (latents_right is None):
            print("------------ WARNING -----------")
            print("LatentBlending problem: Adding noise to latents in diffusion stack left and right of current frame, this shouldn't ever happen...")
            print_info = True

        if latents_left is None:
            print(f"Trying to compute latents at t={args.t_raw:.3f}, index {target_trajectory_index}...")
            print(f"Required latent at t={t_raw_left:.3f}, index {target_trajectory_index} is None")
            #self.print_stack()
            self.construct_noised_latents(args, t_raw_left)
            latents_left = self.latents[t_raw_left][target_trajectory_index]

        if latents_right is None:
            print(f"Trying to compute latents at t={args.t_raw:.3f}, index {target_trajectory_index}...")
            print(f"Required latent at t={t_raw_right:.3f}, index {target_trajectory_index} is None")
            #self.print_stack()
            self.construct_noised_latents(args, t_raw_right)
            latents_right = self.latents[t_raw_right][target_trajectory_index]

        # convert to torch tensors:
        latents_left = torch.from_numpy(latents_left).to(self.device)
        latents_right = torch.from_numpy(latents_right).to(self.device)
        
        return latents_left, latents_right, t_raw_left, t_raw_right

    def print_stack(self):
        import numpy as np
        import pandas as pd

        def compute_std(arr):
            return np.nan if arr is None else arr.std().item()

        data = {}
        for t in sorted(self.latents.keys()):
            data[f"{t:.3f}"] = [compute_std(arr) for arr in self.latents[t]]
        
        df = pd.DataFrame(data)

        new_index = [f"std post {int(i+1):02d} - {int(timestep):03d}" for i, timestep in enumerate(self.pipe.scheduler.timesteps)]
        df.index = [f"std pre  01 - {self.pipe.scheduler.timesteps[0].cpu().item():.0f}"] + new_index
        pd.set_option('display.float_format', '{:.2f}'.format)
        print(df)

    def add_to_phase_data(self, args, t_raw):
        if self.phase_data is None:
            self.phase_data = {key: [] for key in ["t_raw", "init_image_strength", "c", "scale"]}

        self.phase_data['t_raw'].append(t_raw)
        self.phase_data['init_image_strength'].append(args.init_image_strength)
        self.phase_data['c'].append(args.c.cpu().numpy().astype(np.float16))
        self.phase_data['scale'].append(args.guidance_scale)

    def save_to_disk(self):
        phase_data_dir = os.path.join(self.args.frames_dir, "phase_data")
        os.makedirs(phase_data_dir, exist_ok=True)

        self.phase_data["init_image_strength"] = np.array(self.phase_data["init_image_strength"])
        self.phase_data["t_raw"] = np.array(self.phase_data["t_raw"])
        self.phase_data["c"]     = np.array(self.phase_data["c"])
        self.phase_data["scale"] = np.array(self.phase_data["scale"])
        self.phase_data["uc"]    = np.array(self.args.uc.cpu().numpy().astype(np.float16))

        min_t, max_t = min(self.phase_data['t_raw']), max(self.phase_data['t_raw'])
        name_str = f"{int(min_t):03d}_to_{int(max_t):03d}"

        #save the metadata from this phase to disk
        np.savez_compressed(os.path.join(phase_data_dir, name_str), **self.phase_data)
        save_settings(self.args, f'{phase_data_dir}/args.json')

    def load_from_disk(self, path):
        #load the phase data back from disk
        phase_data = np.load(path, allow_pickle=True)
        self.phase_data = {key: phase_data[key] for key in phase_data.files}



from interpolator import perceptual_distance
class FrameBuffer():
    def __init__(self, smooth, args):
        self.smooth = smooth
        self.args = args
        self.clear_buffer(keep_last_frame = False)
        self.current_reduction_f = 0.75  # default multiplier for the perceptual d when creating a new split: old_d = reduction_f * new_d_left + reduction_f * new_d_right
        
    def __len__(self):
        return len(self.frames)

    def clear_buffer(self, keep_last_frame = True):
        if keep_last_frame and len(self) > 0:
            self.frames, self.ts, self.distances = [self.frames[-1]], [0.0], []
        else:
            self.frames, self.ts, self.distances = [], [], []

    def get_max_perceptual_distance(self):
        if len(self.distances) > 0:
            return np.max(self.distances)
        else:
            return -1.0

    def get_perceptual_distance_at_t(self, t):
        if len(self.distances) < 2:
            return 1.0 # default value for random, unrelated images
        index_t_left = np.searchsorted(self.ts, [t%1], side="left")[0]-1
        density_at_t = self.distances[index_t_left]
        return density_at_t

    def add_frame(self, img, t, latents = None):
        self.frames.append(img)
        self.ts.append(t)

        # Sort all the frames according to t (only the last one will be out of order)
        sort_indices = list(np.argsort(self.ts))

        # The last index, is the position where this new frame will go:
        insert_index = sort_indices.index(len(sort_indices)-1)
        self.ts     = [self.ts[i] for i in sort_indices]
        self.frames = [self.frames[i] for i in sort_indices]

        if len(self.frames) >= 2:
            self.update_distances(insert_index)

    def get_t_midpoints(self, use_ts = None):
        if use_ts is None:
            use_ts = np.sorted(self.ts)

        x = []
        for i in range(len(use_ts)-1):
            t_midpoint = (use_ts[i] + use_ts[i+1]) / 2
            x.append(t_midpoint)
        return np.array(x)

    def estimate_density_curve(self, num_points = 100):
        y = np.array(self.distances)
        x = self.get_t_midpoints()

        # given the discrete function samples (x,y), estimate the continuous density curve:
        new_x = np.linspace(np.min(self.ts),np.max(self.ts),num_points) 
        return resample_signal(new_x, x, y)


    def update_distances(self, insert_index):
        if len(self.frames) < 4: # re-compute all the frame distances:
            self.distances = []
            for i in range(len(self.frames)-1):
                self.distances.append(perceptual_distance(self.frames[i],self.frames[i+1]))
        else: # Compute only the two distances for the newly added frame:
            if self.smooth:
                before_distance = perceptual_distance(self.frames[insert_index-1], self.frames[insert_index])
                after_distance  = perceptual_distance(self.frames[insert_index], self.frames[insert_index+1])
                
                old_distance = self.distances[insert_index-1]
                current_reduction_f = ((before_distance + after_distance) / 2) / old_distance

                # moving average:
                self.current_reduction_f = 0.5 * self.current_reduction_f + 0.5 * current_reduction_f
                
                # Remove the old distance from the list and insert the two new distances:
                self.distances.pop(insert_index-1)
                self.distances.insert(insert_index-1, before_distance)
                self.distances.insert(insert_index, after_distance)
            else:
                new_distance = perceptual_distance(self.frames[insert_index-1], self.frames[insert_index])
                self.distances.append(new_distance)

        #print("Distances:", np.round(np.array(self.distances), 3))

    def plot_distances(self, output_dir):
        name_str = f"{time.time():.03f}"
        os.makedirs(output_dir, exist_ok=True)

        if len(self.distances) < 3: return

        midpoint_ts, bar_widths = [], []
        for i in range(len(self.distances)):
            midpoint_ts.append((self.ts[i] + self.ts[i+1]) / 2)
            bar_widths.append(self.ts[i+1] - self.ts[i])
        
        plt.figure(figsize=(24, 10))
        plt.bar(range(len(self.distances)), self.distances, edgecolor='black')
        #plt.bar(midpoint_ts, self.distances, edgecolor='black', linewidth=1, width=bar_widths)
        plt.ylim([0,1])
        plt.title(f"Perceptual distances between {len(self.frames)} frames")
        plt.tight_layout()
        try:
            plt.savefig(os.path.join(output_dir, f'distances_{name_str}.jpg'))
        except:
            plt.savefig(os.path.join(output_dir, f'distances_{name_str}.png'))
        plt.close()

    def get_current_keyframe_imgs(self):
        if len(self) < 2:
            raise ValueError("Frame buffer does not contain two images yet!")
        else:
            return self.frames[0], self.frames[-1]