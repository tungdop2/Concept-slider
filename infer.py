import torch
from PIL import Image
import argparse
import os, json, random
import pandas as pd
import matplotlib.pyplot as plt
import glob, re
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import numpy as np

from safetensors.torch import load_file
import matplotlib.image as mpimg
import copy
import gc
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import DiffusionPipeline
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
    LMSDiscreteScheduler,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor, AttentionProcessor
from typing import Any, Dict, List, Optional, Tuple, Union
from trainscripts.textsliders.lora import (
    LoRANetwork,
    DEFAULT_TARGET_REPLACE,
    UNET_TARGET_REPLACE_MODULE_CONV,
)


def flush():
    torch.cuda.empty_cache()
    gc.collect()
flush()

# path to your model file
lora_weights = [
    "models/age_posneg_slider_alpha1.0_rank4_noxattn/age_posneg_slider_alpha1.0_rank4_noxattn_last.pt",
]
lora_state_dict = torch.load(lora_weights[0])
# wrap it
new_dict = {}
new_dict['lora_state_dict'] = lora_state_dict
new_dict['lora_params'] = {
    "base_model": "stablediffusionapi/realistic-vision-v51",
    "rank": 4,
    "alpha": 1.0,
    "train_method": "noxattn",
    "multiplier": 1.0,
    "network_type": "c3lier",
}
torch.save(new_dict, "models/age_slider.pt")
exit()

output_dir = "test"

pretrained_model_name_or_path = "stablediffusionapi/realistic-vision-v51"

revision = None
device = "cuda:0"
rank = 4
weight_dtype = torch.float16

# Load scheduler, tokenizer and models.
noise_scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    clip_sample=False,
    prediction_type="epsilon"
)
tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path, subfolder="tokenizer", revision=revision
)
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
)
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path, subfolder="vae", revision=revision
)
unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="unet", revision=revision
)
# freeze parameters of models to save more memory
unet.requires_grad_(False)
unet.to(device, dtype=weight_dtype)
vae.requires_grad_(False)

text_encoder.requires_grad_(False)

# For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
# as these weights are only used for inference, keeping weights in full precision is not required.


# Move unet, vae and text_encoder to device and cast to weight_dtype
vae.requires_grad_(False)
vae.to(device, dtype=weight_dtype)
text_encoder.to(device, dtype=weight_dtype)

# Assuming all other imports and required models are loaded as per your original code
prompts = [
    "A selfie of a girl, upper body, smilling, beautiful",
    # "A running girl, full body, sport",
    # "A girl in santa costume, full body",
    # "A woman in a bikini, full body",
    # "A beauty sitting on a chair, full body",
    # "A woman drinking coffee, smiling, upper body",
]
# prompts = [
#     "A woman in tight jeans, full body, viewed from behind.",
#     "A woman in a bikini, standing at the beach, viewed from behind.",
#     "A woman in yoga pants, doing a stretch, viewed from behind.",
#     "A woman in a dress, walking away, viewed from behind."
# ]
scales = [round(x * 0.1, 1) for x in range(-20, 21)]  # Equivalent to range(-1, 3, 0.1)
start_noise = 800
num_images_per_prompt = 1
torch_device = 'cuda'  # Assuming you have a CUDA device, change accordingly
negative_prompt = "gray, blackwhite, nude, naked, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, distorted, ugly"
batch_size = 1
height = 512
width = 512
ddim_steps = 50
guidance_scale = 7.5

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

for prompt in prompts:
    for _ in range(num_images_per_prompt):
        seed = random.randint(0, 1e9)
        for lora_weight in lora_weights:
            if 'full' in lora_weight:
                train_method = 'full'
            elif 'noxattn' in lora_weight:
                train_method = 'noxattn'
            else:
                train_method = 'noxattn'

            network_type = "c3lier"
            if train_method == 'xattn':
                network_type = 'lierla'

            modules = DEFAULT_TARGET_REPLACE
            if network_type == "c3lier":
                modules += UNET_TARGET_REPLACE_MODULE_CONV

            model_name = lora_weight
            name = os.path.basename(model_name)
            unet = UNet2DConditionModel.from_pretrained(
                pretrained_model_name_or_path, subfolder="unet", revision=revision
            )
            unet.requires_grad_(False)
            unet.to(torch_device, dtype=weight_dtype)
            rank = 4
            alpha = 1
            if 'rank4' in lora_weight:
                rank = 4
            if 'rank8' in lora_weight:
                rank = 8
            if 'alpha1' in lora_weight:
                alpha = 1.0

            network = LoRANetwork(
                unet,
                rank=rank,
                multiplier=1.0,
                alpha=alpha,
                train_method=train_method,
            ).to(torch_device, dtype=weight_dtype)
            network.load_state_dict(torch.load(lora_weight))
            images_list = []

            print(prompt, seed)

            for scale in tqdm(scales):
                generator = torch.manual_seed(seed)
                text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

                text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

                max_length = text_input.input_ids.shape[-1]
                if negative_prompt is None:
                    uncond_input = tokenizer(
                        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                    )
                else:
                    uncond_input = tokenizer(
                        [negative_prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                    )
                # Ensure the negative prompt has the same sequence length as the main prompt
                uncond_input_ids = uncond_input.input_ids.to(torch_device)
                if uncond_input_ids.shape[-1] < max_length:
                    padding_length = max_length - uncond_input_ids.shape[-1]
                    uncond_input_ids = torch.nn.functional.pad(uncond_input_ids, (0, padding_length), "constant", 0)
                elif uncond_input_ids.shape[-1] > max_length:
                    uncond_input_ids = uncond_input_ids[:, :max_length]
                
                uncond_embeddings = text_encoder(uncond_input_ids)[0]

                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

                latents = torch.randn(
                    (batch_size, unet.in_channels, height // 8, width // 8),
                    generator=generator,
                )
                latents = latents.to(torch_device)

                noise_scheduler.set_timesteps(ddim_steps)

                latents = latents * noise_scheduler.init_noise_sigma
                latents = latents.to(weight_dtype)
                latent_model_input = torch.cat([latents] * 2)

                for t in noise_scheduler.timesteps:
                    if t > start_noise:
                        network.set_lora_slider(scale=0)
                    else:
                        network.set_lora_slider(scale=scale)

                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep=t)
                    with network:
                        with torch.no_grad():
                            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

                latents = 1 / 0.18215 * latents
                with torch.no_grad():
                    image = vae.decode(latents).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
                images = (image * 255).round().astype("uint8")
                pil_images = [Image.fromarray(image) for image in images]
                pil_images[0].save(f"{output_dir}/{prompt.replace(' ', '_')}_{name.replace('.pt', '')}_{scale}.png")
                images_list.append(pil_images[0])

            del network, unet
            unet = None
            network = None
            torch.cuda.empty_cache()

            # Save images_list as a GIF
            gif_path = os.path.join(output_dir, f"{prompt.replace(' ', '_')}_{name.replace('.pt', '')}.gif")
            images_list[0].save(
                gif_path,
                save_all=True,
                append_images=images_list[1:],
                duration=500,
                loop=0
            )
            print(f"Saved GIF to {gif_path}")