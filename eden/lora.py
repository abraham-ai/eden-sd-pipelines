import os
import sys
from pathlib import Path

SD_PATH = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
LORA_PATH = os.path.join(SD_PATH, 'lora')
LORA_DIFFUSION_PATH = os.path.join(LORA_PATH, 'lora')
sys.path.append(LORA_PATH)
sys.path.append(LORA_DIFFUSION_PATH)

import itertools
import math
import json
import time
import numpy as np
from typing import Optional, List, Literal

import wandb
import torch
import torch.nn.functional as F
import torch.optim as optim
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from cli_lora_pti import *

from lora_diffusion import *

# from lora_diffusion import (
#     PivotalTuningDatasetCapation,
#     extract_lora_ups_down,
#     inject_trainable_lora,
#     inject_trainable_lora_extended,
#     inspect_lora,
#     save_lora_weight,
#     save_all,
#     prepare_clip_model_sets,
#     evaluate_pipe,
#     UNET_EXTENDED_TARGET_REPLACE,
#     parse_safeloras_embeds,
#     apply_learned_embed_in_clip,
#     tune_lora_scale, 
#     patch_pipe, 
#     monkeypatch_or_replace_safeloras, 
#     monkeypatch_remove_lora, 
#     dict_to_lora, 
#     load_safeloras_both, 
#     apply_learned_embed_in_clip, 
#     parse_safeloras, 
#     monkeypatch_or_replace_lora_extended, 
#     parse_safeloras_embeds
# )




def train_lora(
    instance_data_dir: str,
    pretrained_model_name_or_path: str,
    output_dir: str,
    train_text_encoder: bool = True,
    pretrained_vae_name_or_path: str = None,
    revision: Optional[str] = None,
    perform_inversion: bool = True,
    use_template: Literal[None, "object", "style", "person"] = None,
    train_inpainting: bool = False,
    placeholder_tokens: str = "",
    placeholder_token_at_data: Optional[str] = None,
    initializer_tokens: Optional[str] = None,
    load_pretrained_inversion_embeddings_path: Optional[str] = None,
    seed: int = 42,
    resolution: int = 512,
    color_jitter: bool = True,
    train_batch_size: int = 1,
    sample_batch_size: int = 1,
    max_train_steps_tuning: int = 1000,
    max_train_steps_ti: int = 1000,
    save_steps: int = 100,
    gradient_accumulation_steps: int = 4,
    gradient_checkpointing: bool = False,
    lora_rank_unet: int = 4,
    lora_rank_text_encoder: int = 4,
    lora_unet_target_modules={"CrossAttention", "Attention", "GEGLU"},
    lora_clip_target_modules={"CLIPAttention"},
    lora_dropout_p: float = 0.0,
    lora_scale: float = 1.0,
    use_extended_lora: bool = False,
    clip_ti_decay: bool = True,
    learning_rate_unet: float = 1e-4,
    learning_rate_text: float = 1e-5,
    learning_rate_ti: float = 5e-4,
    continue_inversion: bool = False,
    continue_inversion_lr: Optional[float] = None,
    use_face_segmentation_condition: bool = False,
    cached_latents: bool = True,
    use_mask_captioned_data: bool = False,
    mask_temperature: float = 1.0,
    scale_lr: bool = False,
    lr_scheduler: str = "linear",
    lr_warmup_steps: int = 0,
    lr_scheduler_lora: str = "linear",
    lr_warmup_steps_lora: int = 0,
    weight_decay_ti: float = 0.00,
    weight_decay_lora: float = 0.001,
    use_8bit_adam: bool = False,
    device="cuda:0",
    extra_args: Optional[dict] = None,
    log_wandb: bool = False,
    wandb_log_prompt_cnt: int = 10,
    wandb_project_name: str = "new_pti_project",
    wandb_entity: str = "new_pti_entity",
    proxy_token: str = "person",
    enable_xformers_memory_efficient_attention: bool = False,
    out_name: str = "final_lora",
):
    script_start_time = time.time()
    torch.manual_seed(seed)

    if use_template == "person" and not use_face_segmentation_condition:
        print("###  WARNING  ### : Using person template without face segmentation condition")
        print("When training people, it is highly recommended to use face segmentation condition!!")

    # Get a dict with all the arguments:
    args_dict = locals()

    if log_wandb:
        wandb.init(
            project=wandb_project_name,
            entity=wandb_entity,
            name=f"steps_{max_train_steps_ti}_lr_{learning_rate_ti}_{instance_data_dir.split('/')[-1]}",
            reinit=True,
            config={
                **(extra_args if extra_args is not None else {}),
            },
        )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    if len(placeholder_tokens) == 0:
        placeholder_tokens = []
        print("PTI : Placeholder Tokens not given, using null token")
    else:
        placeholder_tokens = placeholder_tokens.split("|")
        assert (
            sorted(placeholder_tokens) == placeholder_tokens
        ), f"Placeholder tokens should be sorted. Use something like {'|'.join(sorted(placeholder_tokens))}'"

    if initializer_tokens is None:
        print("PTI : Initializer Tokens not given, doing random inits")
        initializer_tokens = ["<rand-0.017>"] * len(placeholder_tokens)
    else:
        initializer_tokens = initializer_tokens.split("|")

    assert len(initializer_tokens) == len(
        placeholder_tokens
    ), "Unequal Initializer token for Placeholder tokens."

    if proxy_token is not None:
        class_token = proxy_token
    class_token = "".join(initializer_tokens)

    if placeholder_token_at_data is not None:
        tok, pat = placeholder_token_at_data.split("|")
        token_map = {tok: pat}

    else:
        token_map = {"DUMMY": "".join(placeholder_tokens)}

    print("PTI : Placeholder Tokens", placeholder_tokens)
    print("PTI : Initializer Tokens", initializer_tokens)
    print("PTI : Token Map: ", token_map)

    # get the models
    text_encoder, vae, unet, tokenizer, placeholder_token_ids = get_models(
        pretrained_model_name_or_path,
        pretrained_vae_name_or_path,
        revision,
        placeholder_tokens,
        initializer_tokens,
        device=device,
    )

    noise_scheduler = DDPMScheduler.from_config(
        pretrained_model_name_or_path, subfolder="scheduler"
    )

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if enable_xformers_memory_efficient_attention:
        from diffusers.utils.import_utils import is_xformers_available

        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if scale_lr:
        unet_lr = learning_rate_unet * gradient_accumulation_steps * train_batch_size
        text_encoder_lr = (
            learning_rate_text * gradient_accumulation_steps * train_batch_size
        )
        ti_lr = learning_rate_ti * gradient_accumulation_steps * train_batch_size
    else:
        unet_lr = learning_rate_unet
        text_encoder_lr = learning_rate_text
        ti_lr = learning_rate_ti

    train_dataset = PivotalTuningDatasetCapation(
        instance_data_root=instance_data_dir,
        token_map=token_map,
        use_template=use_template,
        tokenizer=tokenizer,
        size=resolution,
        color_jitter=color_jitter,
        use_face_segmentation_condition=use_face_segmentation_condition,
        use_mask_captioned_data=use_mask_captioned_data,
        train_inpainting=train_inpainting,
    )

    if train_inpainting:
        assert not cached_latents, "Cached latents not supported for inpainting"

        train_dataloader = inpainting_dataloader(
            train_dataset, train_batch_size, tokenizer, vae, text_encoder
        )
    else:
        train_dataloader = text2img_dataloader(
            train_dataset,
            train_batch_size,
            tokenizer,
            vae,
            text_encoder,
            cached_latents=cached_latents,
        )

    index_no_updates = torch.arange(len(tokenizer)) != -1

    for tok_id in placeholder_token_ids:
        index_no_updates[tok_id] = False

    unet.requires_grad_(False)
    vae.requires_grad_(False)

    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    for param in params_to_freeze:
        param.requires_grad = False

    if cached_latents:
        vae = None

    # STEP 1 : Perform Inversion
    if perform_inversion and not cached_latents and (load_pretrained_inversion_embeddings_path is None):
        preview_training_batch(train_dataloader, "inversion")

        print("PTI : Performing Inversion")
        ti_optimizer = optim.AdamW(
            text_encoder.get_input_embeddings().parameters(),
            lr=ti_lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=weight_decay_ti,
        )

        token_ids_positions_to_update = np.where(index_no_updates.cpu().numpy() == 0)
        print("Training embedding of size", text_encoder.get_input_embeddings().weight[token_ids_positions_to_update].shape)

        lr_scheduler = get_scheduler(
            lr_scheduler,
            optimizer=ti_optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=max_train_steps_ti,
        )

        train_inversion(
            unet,
            vae,
            text_encoder,
            train_dataloader,
            max_train_steps_ti,
            cached_latents=cached_latents,
            accum_iter=gradient_accumulation_steps,
            scheduler=noise_scheduler,
            index_no_updates=index_no_updates,
            optimizer=ti_optimizer,
            lr_scheduler=lr_scheduler,
            save_steps=save_steps,
            placeholder_tokens=placeholder_tokens,
            placeholder_token_ids=placeholder_token_ids,
            save_path=output_dir,
            test_image_path=instance_data_dir,
            log_wandb=log_wandb,
            wandb_log_prompt_cnt=wandb_log_prompt_cnt,
            class_token=class_token,
            train_inpainting=train_inpainting,
            mixed_precision=False,
            tokenizer=tokenizer,
            clip_ti_decay=clip_ti_decay,
        )

        del ti_optimizer
        print("###############  Inversion Done  ###############")

    elif load_pretrained_inversion_embeddings_path is not None:

        print("PTI : Loading pretrained inversion embeddings..")
        from safetensors.torch import safe_open
        # Load the pretrained embeddings from the lora file:
        safeloras = safe_open(load_pretrained_inversion_embeddings_path, framework="pt", device="cpu")
        #monkeypatch_or_replace_safeloras(pipe, safeloras)
        tok_dict = parse_safeloras_embeds(safeloras)
        apply_learned_embed_in_clip(
                tok_dict,
                text_encoder,
                tokenizer,
                idempotent=True,
            )

    # Next perform Tuning with LoRA:
    if not use_extended_lora:
        unet_lora_params, _ = inject_trainable_lora(
            unet,
            r=lora_rank_unet,
            target_replace_module=lora_unet_target_modules,
            dropout_p=lora_dropout_p,
            scale=lora_scale,
        )
        print("PTI : not use_extended_lora...")
        print("PTI : Will replace modules: ", lora_unet_target_modules)
    else:
        print("PTI : USING EXTENDED UNET!!!")
        lora_unet_target_modules = (
            lora_unet_target_modules | UNET_EXTENDED_TARGET_REPLACE
        )
        print("PTI : Will replace modules: ", lora_unet_target_modules)
        unet_lora_params, _ = inject_trainable_lora_extended(
            unet, r=lora_rank_unet, target_replace_module=lora_unet_target_modules
        )

    #n_optimizable_unet_params = sum([el.numel() for el in itertools.chain(*unet_lora_params)])
    #print("PTI : Number of optimizable UNET parameters: ", n_optimizable_unet_params)

    params_to_optimize = [
        {"params": itertools.chain(*unet_lora_params), "lr": unet_lr},
    ]

    text_encoder.requires_grad_(False)

    if continue_inversion:
        params_to_optimize += [
            {
                "params": text_encoder.get_input_embeddings().parameters(),
                "lr": continue_inversion_lr
                if continue_inversion_lr is not None
                else ti_lr,
            }
        ]
        text_encoder.requires_grad_(True)
        params_to_freeze = itertools.chain(
            text_encoder.text_model.encoder.parameters(),
            text_encoder.text_model.final_layer_norm.parameters(),
            text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        for param in params_to_freeze:
            param.requires_grad = False
    else:
        text_encoder.requires_grad_(False)

    if train_text_encoder:
        text_encoder_lora_params, _ = inject_trainable_lora(
            text_encoder,
            target_replace_module=lora_clip_target_modules,
            r=lora_rank_text_encoder,
        )
        params_to_optimize += [
            {"params": itertools.chain(*text_encoder_lora_params),
                "lr": text_encoder_lr}
        ]

        #n_optimizable_text_Encoder_params = sum( [el.numel() for el in itertools.chain(*text_encoder_lora_params)])
        #print("PTI : Number of optimizable text-encoder parameters: ", n_optimizable_text_Encoder_params)

    lora_optimizers = optim.AdamW(params_to_optimize, weight_decay=weight_decay_lora)

    unet.train()
    if train_text_encoder:
        print("Training text encoder!")
        text_encoder.train()

    lr_scheduler_lora = get_scheduler(
        lr_scheduler_lora,
        optimizer=lora_optimizers,
        num_warmup_steps=lr_warmup_steps_lora,
        num_training_steps=max_train_steps_tuning,
    )
    if not cached_latents: 
        preview_training_batch(train_dataloader, "tuning")

    #print("PTI : n_optimizable_unet_params: ", n_optimizable_unet_params)
    print(f"PTI : has {len(unet_lora_params)} lora")
    print("PTI : Before training:")

    moved = (
        torch.tensor(list(itertools.chain(*inspect_lora(unet).values())))
        .mean().item())
    print(f"LORA Unet Moved {moved:.6f}")


    moved = (
        torch.tensor(
            list(itertools.chain(*inspect_lora(text_encoder).values()))
        ).mean().item())
    print(f"LORA CLIP Moved {moved:.6f}")

    perform_tuning(
        unet,
        vae,
        text_encoder,
        train_dataloader,
        max_train_steps_tuning,
        index_no_updates = index_no_updates,
        cached_latents=cached_latents,
        scheduler=noise_scheduler,
        optimizer=lora_optimizers,
        save_steps=save_steps,
        placeholder_tokens=placeholder_tokens,
        placeholder_token_ids=placeholder_token_ids,
        save_path=output_dir,
        lr_scheduler_lora=lr_scheduler_lora,
        lora_unet_target_modules=lora_unet_target_modules,
        lora_clip_target_modules=lora_clip_target_modules,
        mask_temperature=mask_temperature,
        tokenizer=tokenizer,
        out_name=out_name,
        test_image_path=instance_data_dir,
        log_wandb=log_wandb,
        wandb_log_prompt_cnt=wandb_log_prompt_cnt,
        class_token=class_token,
        train_inpainting=train_inpainting,
    )

    print("###############  Tuning Done  ###############")
    training_time = time.time() - script_start_time
    print(f"Training time: {training_time/60:.1f} minutes")
    args_dict["training_time_s"] = int(training_time)
    args_dict["n_epochs"] = math.ceil(max_train_steps_tuning / len(train_dataloader.dataset))
    args_dict["n_training_imgs"] = len(train_dataloader.dataset)

    # Save the args_dict to the output directory as a json file:
    with open(os.path.join(output_dir, "lora_training_args.json"), "w") as f:
        json.dump(args_dict, f, default=lambda o: '<not serializable>', indent=2)




"""
class LoraBlender():
    # Helper class to blend LORA models on the fly during interpolations

    def __init__(self, lora_scale):
        self.lora_scale = lora_scale
        self.loras_in_memory = {}
        self.embeds_in_memory = {}

    def load_lora(self, lora_path):
        if lora_path in self.loras_in_memory:
            return self.loras_in_memory[lora_path], self.embeds_in_memory[lora_path]
        else:
            print(f" ---> Loading lora from {lora_path} into memory..")
            safeloras = safe_open(lora_path, framework="pt", device=device)
            embeddings = parse_safeloras_embeds(safeloras)
            
            self.loras_in_memory[lora_path] = safeloras
            self.embeds_in_memory[lora_path] = embeddings

            return safeloras, embeddings

    def blend_embeds(self, embeds_1, embeds_2, t):
        # Blend the two dictionaries of embeddings:
        ret_embeds = {}
        for key in set(list(embeds_1.keys()) + list(embeds_2.keys())):
            if key in embeds_1.keys() and key in embeds_2.keys():
                ret_embeds[key] = (1-t) * embeds_1[key] + t * embeds_2[key]
            elif key in embeds_1.keys():
                ret_embeds[key] = embeds_1[key]
            elif key in embeds_2.keys():
                ret_embeds[key] = embeds_2[key]
        return ret_embeds

    def patch_pipe(self, pipe, t, lora1_path, lora2_path):
        print(f" ---> Patching pipe with lora1 = {os.path.basename(os.path.dirname(lora1_path))} and lora2 = {os.path.basename(os.path.dirname(lora2_path))} at t = {t:.2f}")

        # Load the two loras:
        safeloras_1, embeds_1 = self.load_lora(lora1_path)
        safeloras_2, embeds_2 = self.load_lora(lora2_path)

        metadata = dict(safeloras_1.metadata())
        metadata.update(dict(safeloras_2.metadata()))
        
        # Combine / Linear blend the token embeddings:
        blended_embeds = self.blend_embeds(embeds_1, embeds_2, t)

        # Blend the two loras:
        ret_tensor = {}
        for keys in set(list(safeloras_1.keys()) + list(safeloras_2.keys())):
            if keys.startswith("text_encoder") or keys.startswith("unet"):
                tens1 = safeloras_1.get_tensor(keys)
                tens2 = safeloras_2.get_tensor(keys)
                ret_tensor[keys] = (1-t) * tens1 + t * tens2
            else:
                if keys in safeloras_1.keys():
                    tens = safeloras_1.get_tensor(keys)
                else:
                    tens = safeloras_2.get_tensor(keys)
                ret_tensor[keys] = tens

        loras = dict_to_lora(ret_tensor, metadata)

        # Apply this blended lora to the pipe:
        for name, (lora, ranks, target) in loras.items():
            model = getattr(pipe, name, None)
            if not model:
                print(f"No model provided for {name}, contained in Lora")
                continue
            print("Patching model", name, "with LORA")
            monkeypatch_or_replace_lora_extended(model, lora, target, ranks)

        apply_learned_embed_in_clip(
            blended_embeds,
            pipe.text_encoder,
            pipe.tokenizer,
            token=None,
            idempotent=True,
        )

        # Set the lora scale:
        tune_lora_scale(pipe.unet, self.lora_scale)
        tune_lora_scale(pipe.text_encoder, self.lora_scale)

        return blended_embeds
"""
