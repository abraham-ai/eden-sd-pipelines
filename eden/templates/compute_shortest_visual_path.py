import sys
sys.path.append('..')
import os, shutil
import itertools
from PIL import Image
import torch
from torchvision.transforms import ToTensor, ToPILImage

## python -m pip install tsp_solver2
from tsp_solver.greedy import solve_tsp

from settings import _device
from generation import *
from eden_utils import *

def load_images(directory, target_size = int(768*1.5*768)):
    images, image_paths = [], []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(directory, filename))
    
    images = get_uniformly_sized_crops(image_paths, target_size)

    # convert the images to tensors
    image_tensors = [ToTensor()(img).unsqueeze(0) for img in images]

    print(f"Loaded {len(images)} images from {directory}")
    return list(zip(image_paths, image_tensors))

from tqdm import tqdm

def compute_pairwise_lpips(image_tensors):
    pairwise_distances = {}
    num_combinations = len(image_tensors) * (len(image_tensors) - 1) // 2
    progress_bar = tqdm(total=num_combinations, desc="Computing pairwise LPIPS")

    for img1, img2 in itertools.combinations(image_tensors, 2):
        dist = perceptual_distance(img1[1].to(_device), img2[1].to(_device))
        pairwise_distances[(img1[0], img2[0])] = dist
        pairwise_distances[(img2[0], img1[0])] = dist
        progress_bar.update(1)

    progress_bar.close()
    return pairwise_distances

def create_distance_matrix(pairwise_distances, filenames):
    num_images = len(filenames)
    distance_matrix = [[0 for _ in range(num_images)] for _ in range(num_images)]
    for i, img1 in enumerate(filenames):
        for j, img2 in enumerate(filenames):
            if i != j:
                distance_matrix[i][j] = pairwise_distances[(img1, img2)]
    return distance_matrix

def main(directory):
    paths_and_tensors = load_images(directory)
    filenames = [t[0] for t in paths_and_tensors]

    print(f"Computing {len(filenames)**2} pairwise perceptual distances. This may take a while..")
    pairwise_distances = compute_pairwise_lpips(paths_and_tensors)
    distance_matrix = create_distance_matrix(pairwise_distances, filenames)

    print("Solving traveling salesman problem...")
    path_indices = solve_tsp(distance_matrix, optim_steps=6, endpoints=None)
    path = [filenames[idx] for idx in path_indices]

    outdir = os.path.join(directory, "reordered")
    os.makedirs(outdir, exist_ok=True)

    print(f"Saving optimal visual walkthrough to {outdir}")
    for i, index in enumerate(path_indices):
        original_img_path = paths_and_tensors[index][0]
        json_filepath = original_img_path.replace(".jpg", ".json")
        image_pt_tensor = paths_and_tensors[index][1]
        new_name = f"{i:05d}.jpg"

        pil_image = ToPILImage()(image_pt_tensor.squeeze(0))
        pil_image.save(os.path.join(outdir, new_name))

        if os.path.exists(json_filepath):
            shutil.copy(json_filepath, os.path.join(outdir, new_name.replace(".jpg", ".json")))

if __name__ == "__main__":
    '''
    This script takes a directory of images and computes the shortest visual path through them using Traveling Salesman Solver

    requires:
    pip install tsp_solver2
    
    '''
    import argparse
    parser = argparse.ArgumentParser(description="Compute shortest visual path through images in a directory")
    parser.add_argument("directory", type=str, help="Directory containing images")
    args = parser.parse_args()
    main(args.directory)