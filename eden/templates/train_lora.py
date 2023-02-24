import os
import sys
sys.path.append('..')

from lora import train_lora
from settings import LoraMaskingSettings, LoraTrainingSettings


def train_lora_from_folder(training_folder, output_dir):
    mask_args = LoraMaskingSettings(
        files = training_folder,
        output_dir = training_folder + "/train",
    )
    lora_args = LoraTrainingSettings(
        instance_data_dir = training_folder + "/train",
        output_dir = output_dir,
        pretrained_model_name_or_path = "dreamlike-art/dreamlike-photoreal-2.0",
    )
    lora_location = train_lora(mask_args, lora_args)
    return lora_location


if __name__ == "__main__":
    lora_location = train_lora_from_folder(
        "../assets/yaoyao", 
        "../assets/lora/yaoyao"
    )
    print(lora_location)