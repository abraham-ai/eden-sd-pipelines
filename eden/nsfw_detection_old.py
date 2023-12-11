
import os
import pydload
import onnxruntime
import numpy as np
import logging
import numpy as np

from pipe import SD_PATH
nsfw_model_folder = os.path.join(SD_PATH, 'models/nsfw')

from PIL import Image as pil_image

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        "nearest": pil_image.NEAREST,
        "bilinear": pil_image.BILINEAR,
        "bicubic": pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, "HAMMING"):
        _PIL_INTERPOLATION_METHODS["hamming"] = pil_image.HAMMING
    if hasattr(pil_image, "BOX"):
        _PIL_INTERPOLATION_METHODS["box"] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, "LANCZOS"):
        _PIL_INTERPOLATION_METHODS["lanczos"] = pil_image.LANCZOS



# loads an image and performs the transformations 
def load_image(path, dtype="float32"):
    img = pil_image.open(path)
    size = 240
    img = img.resize((size,size))
    x = np.asarray(img, dtype=dtype)
    x = x.transpose(2, 0, 1)
    x /= 255
    mean = np.array([0.485, 0.456, 0.406], dtype=dtype).reshape(-1,1,1)
    std = np.array([0.229, 0.224, 0.225], dtype=dtype).reshape(-1,1,1)
    x_normalized = (x - mean) / std
    return x_normalized

def load_images(image_paths, image_size, image_names):
    """
    Function for loading images into numpy arrays for passing to model.predict
    inputs:
        image_paths: list of image paths to load
        image_size: size into which images should be resized
    
    outputs:
        loaded_images: loaded images on which keras model can run predictions
        loaded_image_indexes: paths of images which the function is able to process
    
    """
    loaded_images = []
    loaded_image_paths = []

    for i, img_path in enumerate(image_paths):
        try:
            image = load_image(img_path)
            loaded_images.append(image)
            loaded_image_paths.append(image_names[i])
        except Exception as ex:
            logging.exception(f"Error reading {img_path} {ex}", exc_info=True)

    return np.asarray(loaded_images), loaded_image_paths

# sigmoid function
def sig(x):
    return 1/(1 + np.exp(-x))

class Model:
    """
    https://github.com/gsarridis/NSFW-Detection-Pytorch/
    Class for loading model and running predictions.
    For example on how to use take a look the if __name__ == '__main__' part.
    """

    nsfw_model = None

    def __init__(self):
        """
        model = Classifier()
        """
        url = "https://github.com/gsarridis/NSFW-Detection-Pytorch/releases/download/pretrained_models_v2/2022_06_20_11_01_42.onnx"
        url = "https://edenartlab-lfs.s3.amazonaws.com/models/2022_06_20_11_01_42.onnx"

        if not os.path.exists(nsfw_model_folder):
            os.mkdir(nsfw_model_folder)

        model_path = os.path.join(nsfw_model_folder, os.path.basename(url))

        if not os.path.exists(model_path):
            print("Downloading the checkpoint to", model_path)
            pydload.dload(url, save_to_path=model_path, max_time=None)

        self.nsfw_model = onnxruntime.InferenceSession(model_path)


    def predict(
        self,
        image_paths=[],
        batch_size=4,
        image_size=(240, 240)
    ):
        """
        inputs:
            image_paths: list of image paths or can be a string too (for single image)
            batch_size: batch_size for running predictions
            image_size: size to which the image needs to be resized
            categories: since the model predicts numbers, categories is the list of actual names of categories
        """
        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        loaded_images, loaded_image_paths = load_images(
            image_paths, image_size, image_names=image_paths
        )

        if not loaded_image_paths:
            return {}

        preds = []
        model_preds = []
        sigmoid_v = np.vectorize(sig)
        while len(loaded_images):
            _model_preds = self.nsfw_model.run(
                [self.nsfw_model.get_outputs()[0].name],
                {self.nsfw_model.get_inputs()[0].name: loaded_images[:batch_size]},
            )[0]
            _model_preds = sigmoid_v(_model_preds)

            model_preds = [*model_preds, *(np.transpose(_model_preds).tolist()[0])]
            t_preds = np.rint(_model_preds)
            t_preds = np.transpose(t_preds).astype(int).tolist()[0]
            preds = [*preds, *t_preds]

            loaded_images = loaded_images[batch_size:]


        images_preds = {}

        for i, loaded_image_path in enumerate(loaded_image_paths):
            if not isinstance(loaded_image_path, str):
                loaded_image_path = i

            images_preds[loaded_image_path] = {}
            if preds[i]> 0.5:
                images_preds[loaded_image_path] = { 'Label': 'NSFW', 'Score': model_preds[i]}
            else:
                images_preds[loaded_image_path] = { 'Label': 'SFW', 'Score': model_preds[i]}
        return images_preds


if __name__ == "__main__":

    # initialize the model
    net = Model()

    from tqdm import tqdm
    for i in tqdm(range(100)):
        output = net.predict("/data/xander/Projects/cog/eden-sd-pipelines/tmp.jpg")
    print(output)

    # make multiple predictions
    #output = net.predict([<imagepath>, <imagepath>])

"""

import autokeras as ak
from tensorflow.keras.models import load_model
import clip_model
from PIL import Image
import requests
import torch

model_dir_l14 = 'clip_autokeras_binary_nsfw'
model_dir_b32 = 'clip_autokeras_nsfw_b32/'

nsfw_model = None

def setup_nsfw_model(clip):
    global nsfw_model
    model_dir = model_dir_l14 if clip == "ViT-L/14" else model_dir_b32
    nsfw_model = load_model(model_dir, custom_objects=ak.CUSTOM_OBJECTS)
        
def main():
    clip_model.setup(["ViT-L/14"])
    setup_nsfw_model("ViT-L/14")
    url = "https://cdnb.artstation.com/p/assets/images/images/032/142/769/large/ignacio-bazan-lazcano-book-4-final.jpg"
    image = clip_model.load_image_path_or_url(url)
    image_features = clip_model.encode_image(image, "ViT-L/14")
    pred = nsfw_model.predict(image_features.cpu().numpy(), batch_size=1)
    print(pred)

if __name__ == "__main__":
    main()

"""