#!/bin/bash

# Create directory to store logs if it doesn't exist
# Directory for logs
LOG_DIR="logs"

# Clear the logs directory if it exists
if [ -d "$LOG_DIR" ]; then
    rm -rf "$LOG_DIR"/*
fi

# Create the logs directory
mkdir -p "$LOG_DIR"

# List of endpoint descriptions for better clarity in the summary
endpoints=("generate1" "remix1" "upscale1" "controlnet1" "controlnet2" "interrogate1" "lerp1" "lerp2" "real2real1" "blend1")

# Function to run cog predict commands
run_cog_predict() {
    local index=$1
    echo "Running test ${endpoints[$index]}..."
    
    eval "cog predict $2 > $LOG_DIR/${endpoints[$index]}.log 2> $LOG_DIR/${endpoints[$index]}.err"
    local status=$?
    
    if [[ $status -eq 0 ]]; then
        echo "${endpoints[$index]} completed without error."
    else
        echo "${endpoints[$index]} completed with error:"
        cat "$LOG_DIR/${endpoints[$index]}.err"
    fi

    echo "############################################################################"
    
    return $status
}

# Run each cog predict command and store the exit status
commands=(
    '-i mode=generate -i text_input="A ship" -i width=512 -i height=512 -i steps=30 -i n_samples=2 -i upscale_f=1.2'
    '-i mode=remix    -i text_input="A ship" -i width=512 -i height=512 -i steps=30 -i init_image_data="https://generations.krea.ai/images/3cd0b8a8-34e5-4647-9217-1dc03a886b6a.webp"'
    '-i mode=upscale -i init_image_strength=0.65 -i init_image_data="https://generations.krea.ai/images/3cd0b8a8-34e5-4647-9217-1dc03a886b6a.webp" -i width=1280 -i height=1280 -i n_samples=1'
    '-i mode=generate -i text_input="A ship" -i width=512 -i height=512 -i steps=30 -i upscale_f=1.5 -i controlnet_type="canny-edge" -i init_image_data="https://generations.krea.ai/images/3cd0b8a8-34e5-4647-9217-1dc03a886b6a.webp" -i init_image_strength=0.9'
    '-i mode=generate -i text_input="A ship" -i width=512 -i height=512 -i steps=30 -i upscale_f=1.0 -i controlnet_type="canny-edge" -i init_image_data="https://minio.aws.abraham.fun/creations-stg/0f6902d5e83be1f1a3b405b7dfe024ea2b4ed2b2efc07e2851d7301403ca5645.webp" -i init_image_strength=1.0'
    '-i mode=interrogate -i text_input="interrogate" -i init_image_data="https://generations.krea.ai/images/3cd0b8a8-34e5-4647-9217-1dc03a886b6a.webp"'
    '-i mode=interpolate -i text_input="lerp" -i interpolation_seeds="5|8|9" -i interpolation_texts="explosion|cat|8" -i seed=3 -i width=512 -i height=512 -i n_film=1 -i n_frames=12 -i smooth=0 -i loop=1 -i stream=1 -i stream_every=1'
    '-i mode=interpolate -i text_input="lerp" -i interpolation_seeds="5|8|9" -i interpolation_texts="explosion|cat|8" -i seed=3 -i width=512 -i height=512 -i n_film=1 -i n_frames=12 -i smooth=1 -i loop=0 -i stream=0'
    '-i mode=real2real -i text_input="real2real" -i interpolation_seeds="5|8" -i interpolation_init_images="https://generations.krea.ai/images/928271c8-5a8e-4861-bd57-d1398e8d9e7a.webp|https://generations.krea.ai/images/865142e2-8963-47fb-bbe9-fbe260271e00.webp" -i seed=4 -i width=512 -i height=512 -i n_film=1 -i n_frames=12 -i smooth=1 -i loop=1'
    '-i mode=blend -i interpolation_init_images="https://generations.krea.ai/images/928271c8-5a8e-4861-bd57-d1398e8d9e7a.webp|https://generations.krea.ai/images/865142e2-8963-47fb-bbe9-fbe260271e00.webp" -i seed=4 -i width=1024 -i height=1024'
)

statuses=()
for i in "${!commands[@]}"; do
    run_cog_predict $i "${commands[$i]}"
    statuses+=($?)
done

# Analyze the logs
for i in "${!endpoints[@]}"; do 
    if [[ ${statuses[$i]} -eq 0 ]]; then
        echo "########################################################################################"
        echo "${endpoints[$i]} ran successfully"
        echo "########################################################################################"
    else
        echo "########################################################################################"
        echo "########################################################################################"
        echo "${endpoints[$i]} had an error:"
        cat "$LOG_DIR/${endpoints[$i]}.err"
        echo "########################################################################################"
        echo "########################################################################################"
    fi
done
