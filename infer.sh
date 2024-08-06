python infer.py \
    --lora_weight "models/age_slider.pt" \
    --prompts "A selfie of a 30-40 y.o woman, upper body, smiling, beautiful" \
    --output_dir "output" \
    --pretrained_model "stablediffusionapi/realistic-vision-v51" \
    --device "cuda:0" \
    --num_images_per_prompt 5 \
    --negative_prompt "gray, blackwhite, nude, naked, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, distorted, ugly" \
    --batch_size 1 \
    --height 512 \
    --width 512 \
    --ddim_steps 50 \
    --guidance_scale 7.5
