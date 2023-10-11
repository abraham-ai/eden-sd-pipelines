import numpy as np
import torch
import torch.nn as nn
import torchvision
import os, time, math, random
from PIL import Image

from scipy.signal import butter, filtfilt

import matplotlib.pyplot as plt
from generation import *
from einops import rearrange, repeat
from eden_utils import seed_everything, slerp, lerp, create_seeded_noise
from settings import _device

from ip_adapter.ip_adapter import IPAdapterXL

import lpips
lpips_perceptor = lpips.LPIPS(net='alex').eval().to(_device)    # lpips model options: 'squeeze', 'vgg', 'alex'

def tensor_info(img):
    print("Shape: %s, Min: %.3f | Max: %.3f | Mean: %.3f | Std: %.3f" %(str(img.shape), img.min(), img.max(), img.mean(), img.std()))

def prep_pt_img_for_clip(pt_img, clip_preprocessor):
    # This is a bit hacky and can be optimized, but turn the PyTorch img back into a PIL image, since that's what the preprocessor expects:
    pt_img = 255. * rearrange(pt_img, 'b c h w -> b h w c')
    pil_img = Image.fromarray(pt_img.squeeze().cpu().numpy().astype(np.uint8))

    # now, preprocess the image with the CLIP preprocessor:
    clip_img = clip_preprocessor(images=pil_img, return_tensors="pt")["pixel_values"].float().to(_device)
    return clip_img

def resize(img, target_w, mode = "bilinear"):
    b,c,h,w = img.shape
    target_h = int(target_w * h / w)
    resized = torch.nn.functional.interpolate(img, size=(target_h, target_w), mode=mode)
    return resized

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def filter_signal(signal, f_cutoff, dt, plot=False):
    if np.all(signal == signal[0]):
        return signal

    fs = 1 / dt
    b, a = butter_lowpass(f_cutoff, fs)
    y = filtfilt(b, a, signal)
    
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(0, len(signal)*dt, dt), signal, label="Original", alpha=0.5)
        plt.plot(np.arange(0, len(y)*dt, dt), y, label="Filtered", alpha=0.5)
        plt.legend()
        plt.savefig("filtered_signal.png")
        
    return y

@torch.no_grad()
def perceptual_distance(img1, img2, 
    resize_target_pixels_before_computing_lpips = 768):

    '''
    returns perceptual distance between img1 and img2
    This function assumes img1 and img2 are [0,1] and b,c,h,w = img.shape

    By default, images are resized to a fixed resolution before computing the lpips score
    this is useful since some parts of the algorithm are computed from the perceptual distance values.
    '''

    minv1, minv2 = img1.min().item(), img2.min().item()
    minv = min(minv1, minv2)
    if minv < 0:
        print("WARNING: perceptual_distance() assumes images are in [0,1] range.  minv1: %.3f | minv2: %.3f" %(minv1, minv2))

    if resize_target_pixels_before_computing_lpips > 0:
        img1, img2 = resize(img1, resize_target_pixels_before_computing_lpips), resize(img2, resize_target_pixels_before_computing_lpips)

    # lpips model requires image to be in range [-1,1]:
    perceptual_distance = lpips_perceptor((2*img1)-1, (2*img2)-1).mean().item()

    return perceptual_distance





#######################################################################################################
#######################################################################################################


from pipe import update_pipe_with_lora
from planner import FrameBuffer, LatentTracker, resample_signal
class Interpolator():
    '''
    Utility class to interpolate between creations (prompts + seeds + scales)

    Two main modes are smooth=False and smooth=True:
    smooth=False does default, linear stepwise interpolation
    smooth=True will iteratively sample the next interpolation point in the visually densest region
    of the frame sequence by computing perceptual similarity scores between all consecutive frames.

    '''

    def __init__(self, pipe, prompts, n_frames_target, args, device, 
        images = None,
        smooth = True, 
        seeds = None, 
        scales = None, 
        lora_paths = None,
        loop = False):

        self.pipe = pipe
        self.args = args
        self.device = device
        self.prompts, self.seeds, self.scales = prompts, seeds, scales
        if images is not None:
            assert len(images) == len(prompts), "Number of given images must match number of prompts!"
            self.images = images
            self.ip_adapter = IPAdapterXL(pipe, eden_pipe.IP_ADAPTER_IMG_ENCODER_PATH, eden_pipe.IP_ADAPTER_PATH, _device)
        else:
            self.images = [None] * len(prompts)
            self.ip_adapter = None

        self.n = len(self.prompts)        
        self.smooth = smooth

        # Figure out the number of frames per prompt:
        self.n_frames = n_frames_target
        self.n_frames_between_two_prompts = max(1, int((self.n_frames - self.n)/(self.n-1)))
        self.n_frames = int(np.ceil(self.n_frames_between_two_prompts*(self.n-1)+self.n))
        print(f"Rendering {self.n_frames} total frames ({self.n_frames_between_two_prompts} frames between every two prompts)")

        self.interpolation_step, self.prompt_index = 0, 0
        self.ts = np.linspace(0, self.n - 1, self.n_frames)

        # Initialize the frame buffer and latent tracker:
        self.latent_tracker = LatentTracker(self.args, self.pipe, self.smooth)
        self.clear_buffer_at_next_iteration = False
        self.prev_prompt_index = 0

        if self.seeds is None:
            self.seeds = np.random.randint(0, high=9999, size=self.n)
        if self.scales is None:
            self.scales = [args.guidance_scale] * self.n
        
        assert len(self.seeds) == len(self.prompts), "Number of given seeds must match number of prompts!"
        assert len(self.scales) == len(self.prompts), "Number of given scales must match number of prompts!"
        
        # Setup conditioning and noise vectors:
        self.lora_paths = lora_paths
        self.prompt_embeds, self.init_noises = [], []
        self.phase_index = 0
        self.setup_next_creation_conditions(self.phase_index)

    def setup_next_creation_conditions(self, phase_index):
        """
        Setup the all conditioning signals for the next interpolation phase
        """

        if self.lora_paths is not None:
            self.args.lora_path = self.lora_paths[phase_index%len(self.lora_paths)]
            self.pipe = update_pipe_with_lora(self.pipe, self.args)

        if phase_index > 0: # remove the last entries (we might have changed lora_paths)
            self.prompt_embeds.pop()
            self.init_noises.pop()

        for i in range(2):
            index = phase_index + i
            prompt = self.prompts[index]
            image  = self.images[index]
            seed   = self.seeds[index]

            try: # SDXL
                if image is not None:
                    # create the conditioning vectors for the current prompt + image using ip_adapter:
                    prompt_embeds = self.ip_adapter.create_embeds(
                        image, prompt=prompt, negative_prompt=self.args.uc_text, scale=self.args.ip_image_strength
                        )
                else:
                    prompt_embeds = self.pipe.encode_prompt(
                        prompt = prompt,
                        device = self.device,
                        num_images_per_prompt = 1,
                        do_classifier_free_guidance = self.args.guidance_scale > 1.0,
                        negative_prompt = self.args.uc_text
                    )
            except:
                prompt_embeds = self.pipe._encode_prompt(
                    prompt = prompt,
                    device = self.device,
                    num_images_per_prompt = 1,
                    do_classifier_free_guidance = self.args.guidance_scale > 1.0,
                    negative_prompt = self.args.uc_text
                )
                prompt_embeds = [p.unsqueeze(0) for p in prompt_embeds]

            self.prompt_embeds.append(prompt_embeds)
            self.init_noises.append(create_seeded_noise(seed, self.args, self.device))
    
    def get_scale(self, t):
        ''' get the scale for the current frame '''
        try:
            scale = (1-t)*self.scales[self.prompt_index] + t*self.scales[self.prompt_index+1]
        except:
            scale = self.scales[self.prompt_index]
        return scale

    def evaluate_new_t(self, new_t, distance_index, target_perceptual_distances, t_min_treshold, verbose = 1):
        """ given the current perceptual distances in the frames_buffer, 
            evaluate  the "fitness" of the newly proposed split location

            Algo:
            - grab the current perceptual distance of the new split location from the buffer
            - for each possible split location, estimate the new distances
            - compute the total curve mse the new distances and the target density curve
            - return the mse

            - new_t: the new split location
            - target_perceptual_distances: the perceptual distance target curve of the current interpolation phase (either flat, audio driven, or ...)

            TODO: currently the target_perceptual_distances are purely audio driven. But some phases in the interpolation are still slightly 'jumpy' despite LatentBlending,
            maybe there's a way to make the perceptual distance curve more smooth by prefering to put high_perceptual frames on those 'jumpy' parts of the interpolation.
            This could be done by simly tweaking the target_perceptual_distances curve with the locations of those 'jumpy' points t (eg multiplied with a gaussian smoothing kernel of small sigma)
        
        """

        current_distances  = np.array(self.latent_tracker.frame_buffer.distances.copy()) #L

        # Grab the current perceptual distance of the new split location from the buffer:
        current_d = current_distances[distance_index]
        current_d_left = current_distances[distance_index-1] if distance_index > 0 else current_d
        current_d_right = current_distances[distance_index+1] if distance_index < len(current_distances)-1 else current_d
     
        # remove the current distance from the buffer_copy:   
        current_distances = np.delete(current_distances, distance_index)

        # if we were to split in the middle, the new estimated distances would be:
        best_new_t = 0.5*new_t[0] + 0.5*new_t[1]
        best_mse   = np.inf
        best_estimated_perceptual_density_curve = None
        best_predictions = (None, None, None)

        if len(self.latent_tracker.t_raws) < self.args.n_anchor_imgs:
            mixing_fs = np.linspace(0.33,0.66,5) # be more conservative about extreme splits early on
        else:
            mixing_fs = np.linspace(0.20,0.80,10)

        for mixing_f in mixing_fs:

            current_delta_t = new_t[1] - new_t[0]
            current_t_try = (1-mixing_f)*new_t[0] + mixing_f*new_t[1]

            if np.min(np.abs(np.array(self.latent_tracker.frame_buffer.ts) - current_t_try)) < t_min_treshold: # new split location would be too close to existing frames
                #print("Splitting at ", current_t_try, " would be too close to existing ts, skipping!")
                continue

            tmp_distance_copy = current_distances.copy()
            tmp_timepoints_copy = self.latent_tracker.frame_buffer.ts.copy()

            # predict the new distances as a simple linear combination of new_d:
            # TODO replace this with a more sophisticated estimation scheme (eg. spline interpolation) that uses an estimated density curve of the current perceptual distances
            predicted_new_d   = current_d * self.latent_tracker.frame_buffer.current_reduction_f
            predicted_d_left  = mixing_f * 2*predicted_new_d
            predicted_d_right = (1-mixing_f) * 2*predicted_new_d

            # insert the new distances into the buffer_copy:
            tmp_distance_copy = np.insert(tmp_distance_copy, distance_index, predicted_d_right)
            tmp_distance_copy = np.insert(tmp_distance_copy, distance_index, predicted_d_left)

            if 1: # upsample the current distances to the sample rate of the target density curve before computing the MSE:
                # insert current_t_try into the timepoints array: 
                tmp_timepoints_copy = np.insert(tmp_timepoints_copy, distance_index+1, current_t_try)
                tmp_timepoints_copy = np.sort(tmp_timepoints_copy)

                # Given these newly constructed / estimated perceptual distance values, we can compute the upsampled approximation of the perceptual distance curve:
                x     = np.linspace(0,1,len(tmp_distance_copy))
                new_x = np.linspace(0,1,len(target_perceptual_distances)) 
                estimated_perceptual_density_curve = resample_signal(new_x, x, tmp_distance_copy)
            else:
                estimated_perceptual_density_curve = tmp_distance_copy
                
            # Finally, compute the MSE with the target_perceptual_distances:
            estimated_perceptual_density_curve = estimated_perceptual_density_curve / np.mean(estimated_perceptual_density_curve)
            target_perceptual_distances = target_perceptual_distances / np.mean(target_perceptual_distances)
            mse = np.mean(np.abs(estimated_perceptual_density_curve - target_perceptual_distances)**2.0)

            if mse < best_mse:
                best_mse = mse
                best_new_t = current_t_try
                best_estimated_perceptual_density_curve = estimated_perceptual_density_curve
        
        return best_mse, (best_new_t, best_estimated_perceptual_density_curve)


    def find_next_t(self, max_density_diff = 5, verbose = 1):
        """
        --> Use the frame buffer to find the next t value to use for interpolation.
        This implements an interative smoothing algorithm that tries to find the best t value to render the next frame,
        making the final video as visually smooth as possible.

        max_density_diff: how much (visually) denser can one unsplittable region be vs another

        returns the target t for the next frame and a boolean indicating if the current interpolation phase should be ended
        """
        stop = False

        # Determines how close the closest frame can be before trying another location to split
        t_min_treshold = 0.2 / (max_density_diff * self.n_frames_between_two_prompts)
        
        if self.latent_tracker.get_n_frames() == 0: # Render start
            t_midpoint = 0
            return t_midpoint, False
        elif self.latent_tracker.get_n_frames() == 1: # Render end
            t_midpoint = 1.0
            return t_midpoint, False
        elif self.latent_tracker.get_n_frames() == 2: # Render midpoint
            t_midpoint = 0.5
            return t_midpoint, False
        else: # Find the best location to render the next frame:

            perceptual_distances = self.latent_tracker.frame_buffer.distances.copy()

            if self.args.planner is not None:
                #print("###### Using planner to get audio push curve! ######")
                perceptual_target_curve, high_fps_target_curve = self.args.planner.get_audio_push_curve(len(perceptual_distances)+1, self.prompt_index, self.n_frames_between_two_prompts, max_n_samples = self.n_frames_between_two_prompts)
                perceptual_target_curve = high_fps_target_curve

                # filter out high frequencies:
                dt = 1.0 / self.args.fps
                f_cutoff = self.args.fps / 5.0
                perceptual_target_curve = filter_signal(perceptual_target_curve, f_cutoff, dt)
            else:
                perceptual_target_curve = np.ones(len(perceptual_distances)+1)

                if 0:
                    print("####################################################")
                    print("WARNING setting perceptual_target_curve to a sine curve!")
                    print("####################################################")
                    # create a full period sine curve that starts at 0 and ends at 0:
                    # this slows down the keyframes and speeds up the middle frames
                    perceptual_target_curve = np.sin(np.linspace(-np.pi/2, 2*np.pi - np.pi/2, len(perceptual_distances)+1)) + 2.0
                    perceptual_target_curve = perceptual_target_curve / np.mean(perceptual_target_curve)


            # Get all the middle-points between the current timepoints:
            new_ts_to_try = []
            for i in range(len(self.latent_tracker.frame_buffer.ts)-1):
                t_left  = self.latent_tracker.frame_buffer.ts[i]
                t_right = self.latent_tracker.frame_buffer.ts[i+1]
                new_ts_to_try.append((t_left, t_right))

            mse_values, t_datas = [], []
            for distance_index, new_t in enumerate(new_ts_to_try):
                mse, t_data = self.evaluate_new_t(new_t, distance_index, perceptual_target_curve, t_min_treshold, verbose = 0)
                mse_values.append(mse)
                t_datas.append(t_data)

            # Find the best t value to render the next frame at:
            best_fit_mse = np.min(mse_values)
            t_data = t_datas[np.argmin(mse_values)]
            next_t, best_estimated_perceptual_density_curve = t_data

            # plot the current distances / target perceptual curves:
            if self.args.save_distance_data and ((self.latent_tracker.get_n_frames() == self.n_frames_between_two_prompts-1) or (self.latent_tracker.get_n_frames() % 10 == 0)):
                os.makedirs(os.path.join(self.args.frames_dir, "distances"), exist_ok = True)
                plt.figure(figsize = (12,6))
                ts = np.linspace(0,1,len(perceptual_distances))
                plt.bar(ts, perceptual_distances / np.mean(perceptual_distances), label = "current distances (before split)", edgecolor='black', width=1/(len(ts)+1))
                plt.plot(np.linspace(0,1,len(best_estimated_perceptual_density_curve)), best_estimated_perceptual_density_curve, '-o', label = "estimated next perceptual_density_curve", color = "black", linewidth = 2)
                plt.plot(np.linspace(0,1,len(perceptual_target_curve)), perceptual_target_curve, '-o', label = "perceptual_target_curve", color = "red", linewidth = 2)
                plt.legend(loc='upper left')
                plt.title(f"Current distances / target density (interpolation step {self.interpolation_step}, fit_MSE = {best_fit_mse:.3f})")
                plt.ylim([0,4])
                #plt.savefig(os.path.join(os.path.join(self.args.frames_dir, "distances"), "distance_targets_%04d.png" %self.interpolation_step))
                plt.savefig(os.path.join(os.path.join(self.args.frames_dir, "distances"), f"all_distance_targets_{self.phase_index}.png"))
                plt.close('all')
                plt.clf()

                #if best_fit_mse > 0.5:
                #    stop = True

            return next_t, stop

    def reset_buffers(self):
        self.latent_tracker.reset_buffer()
        self.clear_buffer_at_next_iteration = False
        self.args.c = None
        self.phase_index += 1
        self.setup_next_creation_conditions(self.phase_index)

    def get_next_conditioning(self, verbose = 0, save_distances_to_dir = None, t_raw = None):
        '''
        This function should be called iteratively in a loop to yield
        consecutive conditioning signals for the diffusion model

        it returns all the conditioning vectors for creating the next frame
        and returns:
        t: the interpolation step [0 --> 1] between the two active prompts
        t_raw: the total interpolation step in the full sequence of prompts


        TODO: if, during interpolation, we created a frame that increased the perceptual distance at that point,
        remove that frame from the buffer and recompute the interpolation point
        ''' 
        abort = False

        if self.clear_buffer_at_next_iteration:
            self.reset_buffers()

        self.prompt_index = int(self.ts[self.interpolation_step])

        if (self.prompt_index > self.prev_prompt_index): # Last frame of this prompt
            if self.smooth:
                self.clear_buffer_at_next_iteration = True
                self.prev_prompt_index = self.prompt_index
                self.prompt_index -= 1
            else:
                self.reset_buffers()
                self.prev_prompt_index = self.prompt_index

        if self.smooth:
            t, abort = self.find_next_t()
            t_raw = t + self.prompt_index

        else:
            # plot the current distances / target perceptual curves:
            perceptual_distances = self.latent_tracker.frame_buffer.distances.copy()

            if t_raw is None:
                t_raw = self.ts[self.interpolation_step]

            self.prompt_index = int(t_raw)
            t = t_raw % 1

        # Get conditioning signals:
        scale      = self.get_scale(t)
        init_noise = slerp(t, self.init_noises[self.prompt_index], self.init_noises[(self.prompt_index + 1) % self.n], flatten = 1, normalize = 1)
        
        i = self.prompt_index

        try: #sdxl
            p_c   = lerp(t, self.prompt_embeds[i][0], self.prompt_embeds[(i + 1) % self.n][0])
            np_c  = lerp(t, self.prompt_embeds[i][1], self.prompt_embeds[(i + 1) % self.n][1])
            pp_c  = lerp(t, self.prompt_embeds[i][2], self.prompt_embeds[(i + 1) % self.n][2])
            npp_c = lerp(t, self.prompt_embeds[i][3], self.prompt_embeds[(i + 1) % self.n][3])
            prompt_embeds = [p_c, np_c, pp_c, npp_c]
        except: #sd v1 / v2
            p_c   = lerp(t, self.prompt_embeds[i][0], self.prompt_embeds[(i + 1) % self.n][0])
            np_c  = lerp(t, self.prompt_embeds[i][1], self.prompt_embeds[(i + 1) % self.n][1])
            prompt_embeds = [p_c, np_c]
        
        self.interpolation_step += 1
        
        if abort:
            print("Aborting interpolation!")
            # correctly update the step counter before exiting:
            while int(self.ts[self.interpolation_step]) == self.prev_prompt_index:
                self.interpolation_step += 1
            self.clear_buffer_at_next_iteration = True
            self.prev_prompt_index = self.prompt_index
        
        return t, t_raw, prompt_embeds, init_noise, scale, self.prompt_index, abort