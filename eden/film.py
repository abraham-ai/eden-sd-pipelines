import os 
import sys
import shutil
import numpy as np
from pathlib import Path

SD_PATH = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
ROOT_PATH = SD_PATH.parents[0]
FILM_PATH = os.path.join(ROOT_PATH, 'frame-interpolation')
FILM_MODEL_PATH = os.path.join(SD_PATH, '../models/film/film_net/Style/saved_model')
sys.path.append(FILM_PATH)

# avoid tf from allocating all gpu memory:
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
from eval import interpolator_cli
from absl import flags
FLAGS = flags.FLAGS

def interpolate_FILM(frames_dir, times_to_interpolate, max_n_images_per_chunk = 500, remove_orig_files = False, add_prefix=""):

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
        interpolator_cli._run_pipeline()
        
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