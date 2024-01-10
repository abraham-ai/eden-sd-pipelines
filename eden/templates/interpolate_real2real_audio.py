import os, random, sys, tempfile
sys.path.append('..')
from generation import *
from eden_utils import get_random_img_paths_from_dir

if __name__ == "__main__":

    # main render settings
    audio_path = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/audio/versilov/versilov.mp3"
    n_imgs     = 10
    seed       = 1
    
    # Get random images from a directory:
    input_dir  = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/init_imgs/flow"
    output_dir = 'results_real2real_audioreactive_club_long'

    seed_everything(seed)
    img_paths = get_random_img_paths_from_dir(input_dir, n_imgs, sorted=False)

    fin_video_path = real2real_audioreactive(img_paths, audio_path, 
        render_settings = {
            "W": 768,
            "H": 768,
            "steps": 30,
            "seconds_between_keyframes": 10,
            "fps": 10,
        },
        audio_reactivity_settings = {
                'depth_rescale'     : [105., 255.],    # rescale depth map to this range # dont change!!
                '3d_motion_xyz'     : [0.7, 0.7, -90], # camera motion limits
                'circular_motion_period_s': 15,  # the period of the circular xy motion around the center (in seconds)
                '3d_rotation_xyz'   : [0,0,0],
                'brightness_factor' : 0.0003, # 0.001
                'contrast_factor'   : 0.2, #0.4
                'saturation_factor' : 0.3, #0.5
                '2d_zoom_factor'    : 0.00,
                'noise_factor'      : 0.0},
        output_dir = output_dir, seed=seed, debug = 0)