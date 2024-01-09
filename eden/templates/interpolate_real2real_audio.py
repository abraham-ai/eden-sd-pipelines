import os, random, sys, tempfile
sys.path.append('..')
from generation import *
from eden_utils import get_random_img_paths_from_dir

if __name__ == "__main__":

    # main render settings
    audio_path = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/audio/versilov/audio.mp3"
    n_imgs     = 7
    seed       = 1
    
    # Get random images from a directory:
    input_dir  = "/data/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/init_imgs/flow"
    output_dir = 'results_real2real_audioreactive_club_long'

    seed_everything(seed)
    img_paths = get_random_img_paths_from_dir(input_dir, n_imgs, sorted=False)
    fin_video_path = real2real_audioreactive(img_paths, audio_path, output_dir = output_dir, seed=seed, debug = 0)