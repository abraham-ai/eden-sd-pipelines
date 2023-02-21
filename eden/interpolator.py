import numpy as np
import torch
import torch.nn as nn
import os, time, math, random
from PIL import Image

from generation import *
from einops import rearrange, repeat
from eden_utils import seed_everything, slerp, lerp, create_seeded_noise, DataTracker
from settings import _device

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

import torchvision
def resize(img, target_w, mode = "bilinear"):
    b,c,h,w = img.shape
    target_h = int(target_w * h / w)
    resized = torch.nn.functional.interpolate(img, size=(target_h, target_w), mode=mode)
    return resized


@torch.no_grad()
def perceptual_distance(img1, img2, 
    resize_target_pixels_before_computing_lpips = 768):

    '''
    returns perceptual distance between img1 and img2
    This function assumes img1 and img2 are [0,1]

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
        smooth = False, 
        seeds = None, 
        scales = None, 
        scale_modulation_amplitude_multiplier = 0,  # decrease the guidance scale at the midpoint of each keyframe transition
        loop = False):

        self.pipe = pipe
        self.args = args
        self.device = device
        self.prompts, self.seeds, self.scales = prompts, seeds, scales
        self.n = len(self.prompts)        
        self.smooth = smooth
        self.scale_modulation_amplitude_multiplier = scale_modulation_amplitude_multiplier

        # Figure out the number of frames per prompt:
        self.n_frames = n_frames_target
        self.n_frames_between_two_prompts = int((self.n_frames - self.n)/(self.n-1))
        self.n_frames = int(np.ceil(self.n_frames_between_two_prompts*(self.n-1)+self.n))
        print(f"Rendering {self.n_frames} total frames ({self.n_frames_between_two_prompts} frames between every two prompts)")

        self.interpolation_step, self.prompt_index = 0, 0
        self.ts = np.linspace(0, self.n - 1, self.n_frames)

        self.data_tracker = DataTracker()


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
        if self.scale_modulation_amplitude_multiplier > 0:
            print(f"Modulating scale between prompts, scale_modulation_amplitude_multiplier: {self.scale_modulation_amplitude_multiplier:.2f}!")

        # Setup conditioning and noise vectors:
        self.setup_creation_conditions()

    def update_aesthetic_target(self, i):
        """
        Update the aesthetic_target image(s) to finetune the conditioning vector for the current prompt i
        the actual finetuning of the CLIP Text encoder is triggered in ../ldm/modules/encoders/modules.py

        the passed images (self.args.aesthetic_target[i]) should be loaded PIL.Images
        """

        aesthetic_gradients_are_active = (self.args.aesthetic_target is not None) and (self.args.aesthetic_steps != 0) and (self.args.aesthetic_lr != 0.0)
        
        if aesthetic_gradients_are_active:
            if isinstance(self.args.aesthetic_target, str): # the target is a .pt file
                self.model.cond_stage_model.aesthetic_target = self.args.aesthetic_target
            else:
                self.model.cond_stage_model.aesthetic_target = self.args.aesthetic_target[i]
            self.model.cond_stage_model.aesthetic_steps  = self.args.aesthetic_steps

    def setup_creation_conditions(self):
        self.prompt_conditionings, self.init_noises = [], []

        for i in range(len(self.prompts)):
            prompt = self.prompts[i]
            seed   = self.seeds[i]
            self.update_aesthetic_target(i)
            print(f"---- Getting conditioning vector for: \n{prompt}")
            prompt_embeds = self.pipe._encode_prompt(
                prompt,
                self.device,
                1,
                self.args.guidance_scale > 1.0,
                negative_prompt = self.args.uc_text
            )

            uc, c = prompt_embeds[0].unsqueeze(0), prompt_embeds[1].unsqueeze(0)
            self.args.uc = uc
            self.prompt_conditionings.append(c)
            self.init_noises.append(create_seeded_noise(seed, self.args, self.device))
    
    def get_scale(self, t):
        ''' get the scale for the current frame '''
        scale = (1-t)*self.scales[self.prompt_index] + t*self.scales[self.prompt_index+1]
        if self.scale_modulation_amplitude_multiplier > 0:
            # slightly decrease the scale at the midpoint of the transition (t=0.5)
            scale_multiplier = (1 + self.scale_modulation_amplitude_multiplier * abs(t-0.5) - self.scale_modulation_amplitude_multiplier*0.5)
            scale *= scale_multiplier

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
        best_mse = np.inf
        best_estimated_perceptual_density_curve = None
        best_predictions = (None, None, None)
        tracking_data = None

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

                tracking_data = {
                                "predicted_new_d": predicted_new_d,
                                "predicted_d_left": predicted_d_left,
                                "predicted_d_right": predicted_d_right, 
                                "current_t_try": current_t_try,
                                "mixing_f": mixing_f,
                                "current_d": current_d,
                                "current_d_left": current_d_left,
                                "current_d_right": current_d_right,
                                "current_delta_t": current_delta_t,
                                "uid": self.args.uid
                                }
        
        return best_mse, (best_new_t, best_estimated_perceptual_density_curve, tracking_data)


    def find_next_t(self, max_density_diff = 7, verbose = 1):
        """
        --> Use the frame buffer to find the next t value to use for interpolation.
        This implements an interative smoothing algorithm that tries to find the best t value to render the next frame,
        making the final video as visually smooth as possible.

        max_density_diff: how much (visually) denser can one unsplittable region be vs another

        returns the target t for the next frame and a boolean indicating if the current interpolation phase should be ended
        """

        # Determines how close the closest frame can be before trying another location to split
        t_min_treshold = 0.1 / (max_density_diff * self.n_frames_between_two_prompts)
        
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
                perceptual_target_curve, high_fps_target_curve = self.args.planner.get_audio_push_curve(len(perceptual_distances)+1, self.prompt_index, self.n_frames_between_two_prompts, max_n_samples = self.n_frames_between_two_prompts)
                perceptual_target_curve = high_fps_target_curve
            else:
                perceptual_target_curve = np.ones(len(perceptual_distances)+1)

                if 0:
                    # create a full period sine curve that starts at 0 and ends at 0:
                    perceptual_target_curve = np.sin(np.linspace(-np.pi/2, 2*np.pi - np.pi/2, len(perceptual_distances)+1)) + 3.0
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
            next_t, best_estimated_perceptual_density_curve, tracking_data = t_data
            
            self.data_tracker.add(tracking_data)

            # plot the current distances / target perceptual curves:
            if self.args.save_distance_data:
                os.makedirs(os.path.join(self.args.frames_dir, "distances"), exist_ok = True)
                import matplotlib.pyplot as plt
                plt.figure(figsize = (12,6))
                ts = np.linspace(0,1,len(perceptual_distances))
                plt.bar(ts, perceptual_distances / np.mean(perceptual_distances), label = "current distances (before split)", edgecolor='black', width=1/(len(ts)+1))
                plt.plot(np.linspace(0,1,len(best_estimated_perceptual_density_curve)), best_estimated_perceptual_density_curve, '-o', label = "estimated next perceptual_density_curve", color = "black", linewidth = 2)
                plt.plot(np.linspace(0,1,len(perceptual_target_curve)), perceptual_target_curve, '-o', label = "perceptual_target_curve", color = "red", linewidth = 2)
                plt.legend(loc='upper left')
                plt.title(f"Current distances / target density (interpolation step {self.interpolation_step}, fit_MSE = {best_fit_mse:.3f})")
                plt.ylim([0,4])
                plt.savefig(os.path.join(os.path.join(self.args.frames_dir, "distances"), "distance_targets_%04d.png" %self.interpolation_step))
                plt.clf()

            return next_t, False

    def reset_buffers(self):
        print("Resetting buffers! (Clearing data from this interpolation phase)")
        self.latent_tracker.reset_buffer()
        self.clear_buffer_at_next_iteration = False
        self.data_tracker.save()
        self.args.c = None

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
        stop = False

        if self.smooth:
            #self.frame_buffer.maybe_reset()

            if self.clear_buffer_at_next_iteration:
                self.reset_buffers()

            self.prompt_index = int(self.ts[self.interpolation_step])
            if self.prompt_index > self.prev_prompt_index: # Last frame of this prompt
                self.clear_buffer_at_next_iteration = True
                self.prev_prompt_index = self.prompt_index
                self.prompt_index -= 1
            
            t, stop = self.find_next_t()
            t_raw = t + self.prompt_index

            if self.args.save_distance_data is not None and 0: #deprecated
                self.latent_tracker.frame_buffer.plot_distances(os.path.join(self.args.frames_dir, "distances"))

        else:
            if t_raw is None:
                t_raw = self.ts[self.interpolation_step]

            self.prompt_index = int(t_raw)
            t = t_raw % 1

        # Get conditioning signals:
        scale      = self.get_scale(t)
        init_noise = slerp(t, self.init_noises[self.prompt_index], self.init_noises[(self.prompt_index + 1) % self.n], flatten = 1, normalize = 1)
        c          = lerp(t, self.prompt_conditionings[self.prompt_index], self.prompt_conditionings[(self.prompt_index + 1) % self.n])
        #c         = slerp(t, self.prompt_conditionings[self.prompt_index], self.prompt_conditionings[(self.prompt_index + 1) % self.n])

        self.interpolation_step += 1

        if stop:
            # correctly update the step counter before exiting:
            while int(self.ts[self.interpolation_step]) == self.prev_prompt_index:
                self.interpolation_step += 1
            self.clear_buffer_at_next_iteration = True
            self.prev_prompt_index = self.prompt_index
        
        return t, t_raw, c, init_noise, scale, self.prompt_index, self.prompts[self.prompt_index], self.seeds[self.prompt_index]