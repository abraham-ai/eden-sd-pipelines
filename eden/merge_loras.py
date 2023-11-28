import torch
import os
import json
from safetensors import safe_open
from safetensors.torch import save_file, load_file
#from dataset_and_utils import TokenEmbeddingsHandler

# Helper function to read JSON
def read_json_from_path(path):
    with open(path, "r") as f:
        return json.load(f)

def load_lora(lora_path, n_txt_encoders=2):
    print(f"Loading LoRA from {lora_path}...")

    token_map = read_json_from_path(os.path.join(lora_path, "special_params.json"))
    training_args = read_json_from_path(os.path.join(lora_path, "training_args.json"))
    tensors = load_file(os.path.join(lora_path, "lora.safetensors"))

    embeddings = {}
    with safe_open(os.path.join(lora_path, "embeddings.pti"), framework="pt", device="cuda") as f:
        for idx in range(n_txt_encoders):
            key = f"text_encoders_{idx}"
            loaded_embeddings = f.get_tensor(key)
            embeddings[key] = loaded_embeddings
            print(f"embeddings[{key}].shape = {loaded_embeddings.shape}")

    return token_map, training_args, tensors, embeddings

def _load_embeddings(loaded_embeddings, tokenizer, text_encoder):
    # Assuming new tokens are of the format <s_i>
    inserting_toks = [f"<s{i}>" for i in range(loaded_embeddings.shape[0])]
    print(inserting_toks)

    special_tokens_dict = {"additional_special_tokens": inserting_toks}
    tokenizer.add_special_tokens(special_tokens_dict)
    text_encoder.resize_token_embeddings(len(tokenizer))

    train_ids = tokenizer.convert_tokens_to_ids(inserting_toks)
    assert train_ids is not None, "New tokens could not be converted to IDs."

    text_encoder.text_model.embeddings.token_embedding.weight.data[
        train_ids
    ] = loaded_embeddings.to(device="cuda")#.to(dtype=dtype)

def merge_loras(lora_path1, lora_path2, save_path, merge_alpha = 0.5, n_txt_encoders = 2):
    os.makedirs(save_path, exist_ok=True)

    token_map1, training_args1, tensors1, embeddings1 = load_lora(lora_path1, n_txt_encoders)
    token_map2, training_args2, tensors2, embeddings2 = load_lora(lora_path2, n_txt_encoders)

    # 1. Create the merged token map:
    merged_token_map = {"TOK1": "<s0><s1>", "TOK2": "<s2><s3>"}

    # save the merged token map:
    with open(os.path.join(save_path, "special_params.json"), "w") as f:
        json.dump(merged_token_map, f)

    # 2. Create merged embeddings:
    merged_embeddings = {}
    for txt_encoder_key in embeddings1.keys():
        merged_embeddings[txt_encoder_key] = torch.cat([embeddings1[txt_encoder_key], embeddings2[txt_encoder_key]], dim=0)
    # save the merged embeddings:
    save_file(merged_embeddings, os.path.join(save_path, "embeddings.pti"))

    # 3. Create merged tensors:
    merged_tensors = {}
    for key in tensors1.keys():
        if key in tensors2: # Average the tensors using merge_alpha
            merged_tensors[key] = (tensors1[key] * merge_alpha) + (tensors2[key] * (1 - merge_alpha))
        else:
            print(f"Warning: {key} not found in second model.")
            merged_tensors[key] = tensors1[key]

    # Saving merged tensors
    save_file(merged_tensors, f"{save_path}/lora.safetensors")

    print(f"\nMerged LoRa with alpha = {merge_alpha} saved to {save_path}.")


if __name__ == '__main__':
    
    lora_path1 = "/home/rednax/Downloads/lora_combinez/max"
    lora_path2 = "/home/rednax/Downloads/lora_combinez/xander"
    save_path = "/home/rednax/Downloads/lora_combinez/combined"
    merge_loras(lora_path1, lora_path2, save_path, merge_alpha = 0.5)