import numpy as np
import torch
import torch.nn as nn
import os, time, math
from PIL import Image, ImageEnhance

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 15

from audio import create_audio_features
from eden_utils import pil_img_to_latent, slerp, create_seeded_noise, save_settings


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

    def morph_image(self, image, frame_index = None):
        if frame_index is None:
            frame_index = self.frame_index

        # increase the brightness of the init_img:
        enhancer = ImageEnhance.Brightness(image)
        factor = 1 + 0.005 * self.fps_adjusted_percus_features[2, frame_index]
        image = enhancer.enhance(factor)

        # increase the contrast of the init_img:
        enhancer = ImageEnhance.Contrast(image)
        factor = 1 + 0.5 * self.fps_adjusted_percus_features[1, frame_index]
        image = enhancer.enhance(factor)

        # increase the saturation of the init_img:
        enhancer = ImageEnhance.Color(image)
        factor = 1 + 0.5 * self.fps_adjusted_percus_features[1, frame_index]
        image = enhancer.enhance(factor)

        # slightly crop and zoom in on the init_img:
        zoom_factor = 1 + 0.007 * self.fps_adjusted_percus_features[0, frame_index]
        # get the center pixel coordinates:
        x, y = image.size[0]//2, image.size[1]//2
        image = zoom_at(image, x, y, zoom_factor)

        # slightly rotate the init_img:
        # rotation_angle = 0.5 * self.fps_adjusted_percus_features[2, self.frame_index]
        # image = image.rotate(rotation_angle)

        # add noise to the init_img:
        # TODO
        
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





def blend_inits(init1, init2, t, args, real2real = True, anti_shinethrough_power = 1.5):
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
    
    if isinstance(init1, Image.Image):
        blended_init = (left_strength * np.array(init1) + right_strength * np.array(init2)) / (left_strength + right_strength)
        blended_init = Image.fromarray(blended_init.astype(np.uint8))
    else:
        blended_init = slerp(mixing_t, init1, init2, flatten = 1, normalize = 1)
    return blended_init, init_image_strength

def create_blended_init_latent_from_init_img():
    """
    
    This function combines two ideas: rendering an image using an init_img and LatentBlending
    Main workflow:
    1. Create the init_latent using the init_img at the target args.init_image_strength
    2. Grab the surrounding latent vectors from the latent_tracker, also at the target args.init_image_strength
    3. Create a linear combination of the standard init_latent and the surrounding latent vectors
    
    TODO
    """
    return None


def create_init_latent(args, t, init_img0, init_img1, device, pipe, 
    key_latent0 = None,
    key_latent1 = None,
    real2real = True):

    """
    This function is derived from the latent-blending idea:
    https://twitter.com/j_stelzer/status/1613179427659169792

    Instead of creating an init_image, directly create the init_latent, which is a
    linear combination of surrounding latent_vectors (at the same timepoint in the diffusion process)
    higher skip_f values will lead to smoother video transitions and lower render time, but also have less interesting transitions

    """
    latent_tracker = args.interpolator.latent_tracker

    # Project init images into latent space:
    if key_latent0 is None:
        key_latent0 = pil_img_to_latent(init_img0, args, device, pipe)
    if key_latent1 is None:
        key_latent1 = pil_img_to_latent(init_img1, args, device, pipe)

    init_image, init_latent  = None, None
    
    if len(args.interpolator.latent_tracker.t_raws) < args.n_anchor_imgs or (args.latent_blending_skip_f is None):
        # simply alpha-blend the keyframe latents using t:
        init_latent, init_image_strength = blend_inits(key_latent0, key_latent1, t, args, real2real = real2real)
        args.interpolator.latent_tracker.current_init_image_strength = init_image_strength
        latent_tracker.latent_blending_skip_f = 0.0

        if init_image_strength == 1: # first or last frame of the interpolation
            init_latent = None
            init_image = init_img0 if t < 0.5 else init_img1

    else: # Apply Latent-Blending trick:
        alpha_blended_init_latent, init_image_strength = blend_inits(key_latent0, key_latent1, t, args, real2real = real2real)
        
        # apply latent_blending_skip_f to the init_image_strength:
        # this essentially skips-ahead in the denoising process
        # compute latent_blending_skip_f based on the perceptual distance of the current transition (which we'll split in two) 
        # get the perceptual distance at the current transtion that we're going to split:
        current_lpips_distance = latent_tracker.frame_buffer.get_perceptual_distance_at_t(t)

        # We want a gradual increase of skip_f from min_v when current_lpips_distance >= max_d to max_v when current_lpips_distance < min_d
        min_skip_f, max_skip_f = args.latent_blending_skip_f[0], args.latent_blending_skip_f[1]
        # hardcoded lpips_perceptual distances corresponding to min_v and max_v latent_skip_f values:
        if args.interpolator.n_frames_between_two_prompts > 64 and 0: # TODO make this autotuning
            max_d, min_d = 0.4, 0.075 # long renders (loads of frames)
        else:
            max_d, min_d = 0.6, 0.1 # normal renders
        latent_tracker.latent_blending_skip_f = min_skip_f + (max_skip_f - min_skip_f) * (max_d - current_lpips_distance) / (max_d - min_d)
        latent_tracker.latent_blending_skip_f = np.clip(latent_tracker.latent_blending_skip_f, min_skip_f, max_skip_f)

        init_image_strength = init_image_strength * (1.0 - latent_tracker.latent_blending_skip_f) + latent_tracker.latent_blending_skip_f
        
        # adjust the noise level of the alpha-blended init_latent:
        # alpha_blended_init_latent = latent_tracker.adjust_denoised_latent_noise_level(alpha_blended_init_latent, init_image_strength)
        
        # grab the nearest neighbouring latents, at the same timepoint in the diffusion process:
        latent_left, latent_right, t_raw_left, t_raw_right = latent_tracker.get_neighbouring_latents(args, adjusted_init_image_strength = init_image_strength)

        # linearly interpolate between the neighbouring latents for create the new init_latent:
        mixing_f = (t - t_raw_left%1) / (t_raw_right - t_raw_left)
        init_latent = (1-mixing_f) * latent_left + mixing_f * latent_right

        # store the init_latent in the latent_tracker: this skips adding latent noise in generate(args) sampler schedules
        latent_tracker.force_starting_latent = init_latent

    return init_latent, init_image, init_image_strength



def get_k_sigmas(pipe, init_image_strength, steps):
    pipe.scheduler.set_timesteps(steps, device="cuda")

    # TODO: This function will error for schedulers that dont have sigmas

    # Compute the number of remaining denoising steps:
    t_enc = int((1.0-init_image_strength) * steps)

    # Noise schedule for the k-diffusion samplers:
    k_sigmas_full = pipe.scheduler.sigmas

    # Extract the final sigma-noise levels to use for denoising:
    k_sigmas = k_sigmas_full[len(k_sigmas_full)-t_enc-1:]

    return k_sigmas, k_sigmas_full




class LatentTracker():
    """
    Helper class that stores all intermediate diffusion latents and frames to enable Latent-Blending and smoothing
    originally implemented for https://twitter.com/xsteenbrugge/status/1558508866463219712
    further improvements inspired by: https://twitter.com/j_stelzer/status/1613179427659169792
    """
    def __init__(self, args, pipe, smooth):
        self.latents = {}
        self.t_raws = []
        self.pipe = pipe
        self.args = args
        self.force_starting_latent = None
        self.device = "cuda"
        self.steps = args.steps
        self.latent_blending_skip_f = 0.0
        self.frame_buffer = FrameBuffer(smooth, args)
        self.phase_data = None

    def get_n_frames(self):
        return len(self.frame_buffer.frames)

    def reset_buffer(self):
        if self.args.save_phase_data:
            self.save_to_disk()
            self.phase_data = None

        self.force_starting_latent = None
        max_t_raw = np.max(self.t_raws)
        last_frame_latents = self.latents[max_t_raw].copy()
        self.latents = {}
        self.t_raws = []

        # add the last frame latents to the new buffer:
        self.latents[max_t_raw] = last_frame_latents
        self.t_raws.append(max_t_raw)
        
        self.frame_buffer.clear_buffer()

    def create_new_denoising_trajectory(self, args):
        # create a new denoising trajectory for the given timestep
        self.current_t_raw = args.t_raw
        self.current_init_image_strength  = args.init_image_strength
        self.noise = create_seeded_noise(args.seed, args, self.device)
        self.t_raws.append(self.current_t_raw)
        self.latents[self.current_t_raw] = []

    def print_latent_history(self, std_ids = None):
        std_ids = [0,1,2,-2,-1]
        stds = []
        print('---------------------')
        print("t_raws:")
        print(np.round(np.array(self.t_raws), 3))
        for t_raw in self.t_raws:
            std = np.array([latent.std().item() for latent in self.latents[t_raw]])
            if std_ids is not None:
                std = std[std_ids]
            if len(std) > 0:
                stds.append(np.round(std,3))
        if len(stds) > 0:
            stds = np.stack(stds).T
            print(stds.shape)
            print(stds)

    def adjust_denoised_latent_noise_level(self, denoised_latent, init_image_strength):
        # adjust the noise level of a fully denoised latent to match the desired init_image_strength
        current_sigmas, _ = get_k_sigmas(self.pipe, init_image_strength, self.steps)
        noised_latent = denoised_latent + self.noise * current_sigmas[0]
        return noised_latent

    def add_frame(self, args, img_t, t, t_raw):
        self.frame_buffer.add_frame(img_t, t)
        self.add_to_phase_data(args, t_raw)

    def add_latent(self, latent, init_image_strength = None, verbose = 0):

        if init_image_strength is not None:
            self.current_init_image_strength = init_image_strength

        if len(self.latents[self.current_t_raw]) == 0:
            # Given the current init_image_strength, pre-add the full noise trajectory all the way to full noise:
            n_denoising_steps = int((1.0 - self.current_init_image_strength) * self.steps)
            n_latents_to_prepend = self.steps - n_denoising_steps

            # This works correctly for klms and euler, but not for euler_ancestral: why???
            current_sigmas, k_sigmas_full = get_k_sigmas(self.pipe, self.current_init_image_strength, self.steps)
            active_sigmas = k_sigmas_full - current_sigmas[0]

            for i, k_sigma in enumerate(active_sigmas[:n_latents_to_prepend]):
                noised_latent = latent + self.noise * k_sigma
                self.latents[self.current_t_raw].append(noised_latent.clone().detach().cpu().numpy())
        
        # append the actual, current latent at this stage of the denoising trajectory:
        self.latents[self.current_t_raw].append(latent.clone().detach().cpu().numpy())

        if len(self.latents[self.current_t_raw]) == self.steps + 1:
            std_zero = self.latents[self.current_t_raw][0].std()
            if std_zero > 15.1 or std_zero < 13.5:
                print('#####################################')
                print(f"WARNING: std_zero is {std_zero:.4f}!")
                print("This shouldn't happen! Something is wrong with the latent blending.")
                print("This could be fixed by using a different sampler, otherwise ask Xander")
                self.print_latent_history()
            elif verbose:
                self.print_latent_history()

    def get_neighbouring_ts(self, t_raw):
        sorted_t_raws = sorted(self.t_raws)
        # Find the closest t that's smaller than args.t_raw:
        t_raw_left = sorted_t_raws[np.searchsorted(sorted_t_raws, [t_raw], side="left")[0]-1]
        # Find the closest t that's larger than args.t_raw:
        t_raw_right = sorted_t_raws[np.searchsorted(sorted_t_raws, [t_raw], side="right")[0]]

        return t_raw_left, t_raw_right

    def get_neighbouring_latents(self, args, adjusted_init_image_strength = None):
        if adjusted_init_image_strength is not None:
            self.current_init_image_strength = adjusted_init_image_strength

        t_raw_left, t_raw_right = self.get_neighbouring_ts(args.t_raw)

        # Now, let's compute the correct latent index into the denoising trajectory:
        current_trajectory_index = self.steps - int((1.0 - self.current_init_image_strength) * self.steps)
        
        # get the latents for the neighbouring frames:
        latents_left = self.latents[t_raw_left][current_trajectory_index]
        latents_right = self.latents[t_raw_right][current_trajectory_index]

        # compute a perceptual distance metric (L2-norm in latent space) of the neighbouring frames (at the end of the denoising trajectory):
        # l2_dist = np.linalg.norm(self.latents[t_raw_left][-1] - self.latents[t_raw_right][-1])

        # convert to torch tensors:
        latents_left = torch.from_numpy(latents_left).to(self.device)
        latents_right = torch.from_numpy(latents_right).to(self.device)
        return latents_left, latents_right, t_raw_left, t_raw_right

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

                tracking_data = {"actual_d_left": before_distance,
                                "actual_d_right": after_distance,
                                "old_distance": old_distance}

                self.args.interpolator.data_tracker.add(tracking_data)

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