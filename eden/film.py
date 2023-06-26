import os 
import sys
import shutil
import numpy as np
from pathlib import Path
import gc
import argparse

SD_PATH = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
ROOT_PATH = SD_PATH.parents[0]
FILM_PATH = os.path.join(ROOT_PATH, 'frame-interpolation')
FILM_MODEL_PATH = os.path.join(SD_PATH, '../models/film/film_net/Style/saved_model')
sys.path.append(FILM_PATH)

import tensorflow as tf

# avoid tf from allocating all gpu memory:
tf_memory_limit = 1024 * 6 # 12GB
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)  # Enable memory growth
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=tf_memory_limit)])

from absl import flags
FLAGS = flags.FLAGS

def interpolate_FILM(frames_dir, times_to_interpolate, 
    max_n_images_per_chunk = 500, 
    remove_orig_files = False, 
    add_prefix="",
    update_film_model_path = None):

    if update_film_model_path is not None:
        global FILM_MODEL_PATH
        FILM_MODEL_PATH = update_film_model_path

    from eval import interpolator_cli

    allowed_extensions = ['.jpg', '.png', '.jpeg']
    frame_paths = sorted([f for f in os.listdir(frames_dir) if os.path.splitext(f)[1] in allowed_extensions])

    print("--------------------------")
    print("Running FILM interpolation")
    print("Input dir: ", frames_dir)
    print("Interpolating %d image files," %len(frame_paths))
    print("n times to interpolate: ", times_to_interpolate)
    print("--------------------------")    

    if len(frame_paths) > max_n_images_per_chunk:
        n_chunks   = int(np.ceil(len(frame_paths) / max_n_images_per_chunk))
        for i in range(n_chunks):
            print("Interpolating chunk %d/%d" %(i+1, n_chunks))
            chunk_dir = os.path.join(frames_dir, "chunk_%02d" %i)
            os.makedirs(chunk_dir, exist_ok=True)
            end = min(len(frame_paths), (i+1)*max_n_images_per_chunk)
            chunk_files = frame_paths[i*max_n_images_per_chunk:end]
            for filepath in chunk_files:
                if remove_orig_files:
                    shutil.move(os.path.join(frames_dir, filepath), os.path.join(chunk_dir, filepath))
                else:
                    shutil.copy(os.path.join(frames_dir, filepath), os.path.join(chunk_dir, filepath))
            interpolate_FILM(chunk_dir, times_to_interpolate, max_n_images_per_chunk = max_n_images_per_chunk)

            # The interpolated frames of chunk_i are now in chunk_dir/interpolated_frames
            # Move them back to frames_dir, prepending the chunk number to the filename:
            os.makedirs(os.path.join(frames_dir, "interpolated_frames"), exist_ok=True)
            for j, filepath in enumerate(os.listdir(os.path.join(chunk_dir, "interpolated_frames"))):
                if i != 0 and j==0: # skip copying duplicate first frames
                    continue
                shutil.move(os.path.join(chunk_dir, "interpolated_frames", filepath), os.path.join(frames_dir, "interpolated_frames/%s%02d_%s" %(add_prefix, i, filepath)))
            
            shutil.rmtree(chunk_dir)

    else:
        args = ["", 
            "--pattern", frames_dir,
            "--model_path", FILM_MODEL_PATH,
            "--times_to_interpolate", str(times_to_interpolate)]
        FLAGS(args)

        # Run the FILM interpolation with TF:
        interpolator_cli._run_pipeline()

    # delete the pipeline object (so we can clear all gpu memory)
    del interpolator_cli
    # run garbage collection:
    gc.collect()
    # release all tf memory:
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.Session.close()
        
    return os.path.join(frames_dir, "interpolated_frames")


def get_n_interpolate(target_n_frames, n_source_frames = 2):
    """
    When using film frames as inits with smooth=True for the interpolator,
    some areas of the [0,1] range will be more densely sampled, and so we need more
    init film frames than there are final interpolation frames.
    """

    times_to_interpolate = 1
    while (2**times_to_interpolate+1)*(n_source_frames-1) < target_n_frames:
        times_to_interpolate += 1

    return times_to_interpolate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', type=str, help='Root directory of the frames to be interpolated')
    parser.add_argument('--times_to_interpolate', type=int, default=1, help='0,1,2,... how many times to interpolate each frame')
    parser.add_argument('--max_n_images_per_chunk', type=int, default=500, help='How many frames to process at once (default: 500)')
    parser.add_argument('--remove_orig_files', action='store_true', help='Whether to remove the original frames after interpolation')
    parser.add_argument('--add_prefix', type=str, default="", help='Prefix to add to the interpolated frames')
    parser.add_argument('--update_film_model_path', type=str, default=None, help='overwrite the film model path')
    args = parser.parse_args()
    
    output_folder = interpolate_FILM(args.frames_dir, args.times_to_interpolate, args.max_n_images_per_chunk, args.remove_orig_files, args.add_prefix, args.update_film_model_path)
