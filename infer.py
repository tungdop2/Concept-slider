import argparse
import os
import random
import gc
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from trainscripts.textsliders.lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV

def flush():
    torch.cuda.empty_cache()
    gc.collect()

def prepare_base_model(pretrained_model_name_or_path, revision, device, weight_dtype):
    try:
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

        unet.requires_grad_(False)
        unet.to(device, dtype=weight_dtype)
        vae.requires_grad_(False)
        vae.to(device, dtype=weight_dtype)
        text_encoder.requires_grad_(False)
        text_encoder.to(device, dtype=weight_dtype)

        return noise_scheduler, tokenizer, text_encoder, vae, unet
    except Exception as e:
        print(f"Error preparing base model: {e}")
        raise

def load_lora_for_unet(lora_weight, unet, device, weight_dtype):
    try:
        lora_dict = torch.load(lora_weight, map_location="cpu")
        
        lora_params = lora_dict['lora_params']
        print(f"LoRA parameters: {lora_params}")
            
        alpha = lora_params['alpha']
        rank = lora_params['rank']
        train_method = lora_params['train_method']
        multiplier = lora_params['multiplier']
        network_type = lora_params['network_type']
        
        modules = DEFAULT_TARGET_REPLACE
        if network_type == "c3lier":
            modules += UNET_TARGET_REPLACE_MODULE_CONV

        network = LoRANetwork(
            unet,
            rank=rank,
            multiplier=multiplier,
            alpha=alpha,
            train_method=train_method,
        ).to(device, dtype=weight_dtype)
        network.load_state_dict(lora_dict['lora_state_dict'])

        return network
    except Exception as e:
        print(f"Error loading LoRA for UNet: {e}")
        raise

def generate_images(prompt, scales, lora_weight, output_dir, pretrained_model_name_or_path, revision, device, weight_dtype, start_noise, num_images_per_prompt, negative_prompt, batch_size, height, width, ddim_steps, guidance_scale):
    os.makedirs(output_dir, exist_ok=True)
    noise_scheduler, tokenizer, text_encoder, vae, unet = prepare_base_model(pretrained_model_name_or_path, revision, device, weight_dtype)

    for _ in range(num_images_per_prompt):
        seed = random.randint(0, int(1e9))
        
        model_name = lora_weight
        name = os.path.basename(model_name)

        network = load_lora_for_unet(lora_weight, unet, device, weight_dtype)
        images_list = []

        print(f"Prompt: {prompt}")
        print(f"Seed: {seed}")

        scales = ["base"] + scales
        for scale in tqdm(scales):
            generator = torch.manual_seed(seed)
            text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

            max_length = text_input.input_ids.shape[-1]
            if negative_prompt is None:
                uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
            else:
                uncond_input = tokenizer([negative_prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
            
            uncond_input_ids = uncond_input.input_ids.to(device)
            if uncond_input_ids.shape[-1] < max_length:
                padding_length = max_length - uncond_input_ids.shape[-1]
                uncond_input_ids = torch.nn.functional.pad(uncond_input_ids, (0, padding_length), "constant", 0)
            elif uncond_input_ids.shape[-1] > max_length:
                uncond_input_ids = uncond_input_ids[:, :max_length]
            
            uncond_embeddings = text_encoder(uncond_input_ids)[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            latents = torch.randn((batch_size, unet.in_channels, height // 8, width // 8), generator=generator)
            latents = latents.to(device)

            noise_scheduler.set_timesteps(ddim_steps)
            latents = latents * noise_scheduler.init_noise_sigma
            latents = latents.to(weight_dtype)
            latent_model_input = torch.cat([latents] * 2)

            for t in noise_scheduler.timesteps:
                if t > start_noise or scale == "base":
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

        del network
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Generate images using a pre-trained model and LoRA weights.")
    parser.add_argument('--lora_weight', type=str, required=True, help="LORA weights for image generation.")
    parser.add_argument('--scales', type=str, required=True, help="Semicolon-separated list of scales.")  # Note the delimiter change
    parser.add_argument('--prompts', type=str, required=True, help="Comma-separated list of prompts.")
    parser.add_argument('--output_dir', type=str, default='output', help="Directory to save the generated images and GIFs.")
    parser.add_argument('--pretrained_model', type=str, default="stablediffusionapi/realistic-vision-v51", help="Pre-trained model path.")
    parser.add_argument('--revision', type=str, default=None, help="Model revision.")
    parser.add_argument('--device', type=str, default="cuda:0", help="Device to use for computation.")
    parser.add_argument('--weight_dtype', type=str, default="float16", help="Data type for model weights.")
    parser.add_argument('--start_noise', type=int, default=800, help="Start noise for image generation.")
    parser.add_argument('--num_images_per_prompt', type=int, default=1, help="Number of images to generate per prompt.")
    parser.add_argument('--negative_prompt', type=str, default="gray, blackwhite, nude, naked, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, distorted, ugly", help="Negative prompt for image generation.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for image generation.")
    parser.add_argument('--height', type=int, default=512, help="Height of the generated images.")
    parser.add_argument('--width', type=int, default=512, help="Width of the generated images.")
    parser.add_argument('--ddim_steps', type=int, default=50, help="Number of DDIM steps for image generation.")
    parser.add_argument('--guidance_scale', type=float, default=7.5, help="Guidance scale for image generation.")
    
    args = parser.parse_args()
    print(args)
    
    scales = list(map(float, args.scales.split(',')))  # Use semicolon to split scales
    print(f"Scales: {scales}")  # Add this line to print parsed scales
    
    prompts = args.prompts.split(',')
    print(f"Prompts: {prompts}")  # Add this line to print parsed prompts
    
    if not torch.cuda.is_available():
        if "cuda" in args.device:
            print("CUDA is not available. Switching to CPU.")
            args.device = "cpu"
    
    for prompt in prompts:
        generate_images(
            prompt=prompt,
            scales=scales,
            lora_weight=args.lora_weight,
            output_dir=args.output_dir,
            pretrained_model_name_or_path=args.pretrained_model,
            revision=args.revision,
            device=args.device,
            weight_dtype=getattr(torch, args.weight_dtype),
            start_noise=args.start_noise,
            num_images_per_prompt=args.num_images_per_prompt,
            negative_prompt=args.negative_prompt,
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
            ddim_steps=args.ddim_steps,
            guidance_scale=args.guidance_scale
        )

if __name__ == "__main__":
    main()
