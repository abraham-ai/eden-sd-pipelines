
#!/bin/bash


# Define variables

TEXT_INPUT1="a photo of a magic stone portal in the forest"
TEXT_INPUT2="a photo of a massive banana on the summit of a mountain"

INIT_IMAGE_URL1="https://storage.googleapis.com/public-assets-xander/A_workbox/init_imgs/img_00003.jpg"
INIT_IMAGE_URL2="https://storage.googleapis.com/public-assets-xander/A_workbox/init_imgs/img_00005.jpg"
INIT_IMAGE_URL3="https://storage.googleapis.com/public-assets-xander/A_workbox/init_imgs/img_00006.jpg"

LORA_URL="https://storage.googleapis.com/public-assets-xander/A_workbox/marzipan_lora_oct_16.tar"

INIT_IMAGE_STRENGTH1=0.0
INIT_IMAGE_STRENGTH2=0.2
INIT_IMAGE_STRENGTH_CONTROL1=0.4
INIT_IMAGE_STRENGTH_CONTROL2=0.7

STEPS=20
WIDTH=768
HEIGHT=768
N_FRAMES=20

UPSCALE_F=1.25
NSAMPLES=2
SEED=0

###########################################################

# Delete output.log if it exists
[ -e "output.log" ] && rm output.log

# Delete error.log if it exists
[ -e "error.log" ] && rm error.log


################################################


# Define request variables

REQUEST1=$(cat <<- EOM
{
    "input": {
      "mode": "interpolate",
      "interpolation_texts": "$TEXT_INPUT1|$TEXT_INPUT2",
      "init_image_data": "$INIT_IMAGE_URL1",
      "init_image_strength": $INIT_IMAGE_STRENGTH1,
      "steps": $STEPS,
      "width": $WIDTH,
      "height": $HEIGHT,
      "n_frames": $N_FRAMES,
      "n_film": 1,
      "seed": $SEED
    }
}
EOM
)

REQUEST2=$(cat <<- EOM
{
    "input": {
      "mode": "real2real",
      "interpolation_init_images": "$INIT_IMAGE_URL1|$INIT_IMAGE_URL2",
      "lora": "$LORA_URL",
      "steps": $STEPS,
      "width": $WIDTH,
      "height": $HEIGHT,
      "n_frames": $N_FRAMES,
      "seed": $SEED
    }
}
EOM
)

REQUEST3=$(cat <<- EOM
{
    "input": {
      "mode": "real2real",
      "interpolation_init_images": "$INIT_IMAGE_URL1|$INIT_IMAGE_URL2",
      "steps": $STEPS,
      "width": $WIDTH,
      "height": $HEIGHT,
      "n_frames": $N_FRAMES,
      "seed": $SEED
    }
}
EOM
)

REQUEST4=$(cat <<- EOM
{
    "input": {
      "mode": "interpolate",
      "interpolation_texts": "$TEXT_INPUT1|$TEXT_INPUT2",
      "init_image_data": "$INIT_IMAGE_URL1",
      "init_image_strength": $INIT_IMAGE_STRENGTH2,
      "controlnet_type": "luminance",
      "steps": $STEPS,
      "width": $WIDTH,
      "height": $HEIGHT,
      "n_frames": $N_FRAMES,
      "n_film": 1,
      "seed": $SEED
    }
}
EOM
)


REQUEST5=$(cat <<- EOM
{
    "input": {
      "mode": "interpolate",
      "interpolation_texts": "$TEXT_INPUT1|$TEXT_INPUT2",
      "init_image_data": "$INIT_IMAGE_URL1",
      "init_image_strength": $INIT_IMAGE_STRENGTH2,
      "controlnet_type": "depth",
      "lora": "$LORA_URL",
      "steps": $STEPS,
      "width": $WIDTH,
      "height": $HEIGHT,
      "n_frames": $N_FRAMES,
      "n_film": 1,
      "seed": $SEED
    }
}
EOM
)


REQUEST6=$(cat <<- EOM
{
    "input": {
      "mode": "generate",
      "text_input": "$TEXT_INPUT",
      "steps": $STEPS,
      "width": $WIDTH,
      "height": $HEIGHT,
      "seed": $SEED
    }
}
EOM
)

REQUEST7=$(cat <<- EOM
{
    "input": {
      "mode": "generate",
      "text_input": "a photo of <concept> on the beach, drinking Coca-Cola",
      "lora": "$LORA_URL",
      "seed": $SEED
    }
}
EOM
)

REQUEST8=$(cat <<- EOM
{
    "input": {
      "mode": "generate",
      "text_input": "$TEXT_INPUT",
      "steps": $STEPS,
      "width": $WIDTH,
      "height": $HEIGHT,
      "n_samples": $NSAMPLES,
      "seed": $SEED
    }
}
EOM
)

REQUEST9=$(cat <<- EOM
{
    "input": {
      "mode": "generate",
      "text_input": "$TEXT_INPUT",
      "steps": $STEPS,
      "width": $WIDTH,
      "height": $HEIGHT,
      "upscale_f": $UPSCALE_F,
      "seed": $SEED
    }
}
EOM
)

REQUEST10=$(cat <<- EOM
{
    "input": {
      "mode": "generate",
      "text_input": "$TEXT_INPUT",
      "steps": $STEPS,
      "width": $WIDTH,
      "height": $HEIGHT,
      "n_samples": $NSAMPLES,
      "upscale_f": $UPSCALE_F,
      "seed": $SEED
    }
}
EOM
)

REQUEST11=$(cat <<- EOM
{
    "input": {
      "mode": "generate",
      "text_input": "$TEXT_INPUT",
      "init_image_data": "$INIT_IMAGE_URL1",
      "init_image_strength": $INIT_IMAGE_STRENGTH1,
      "seed": $SEED
    }
}
EOM
)

REQUEST12=$(cat <<- EOM
{
    "input": {
      "mode": "generate",
      "text_input": "$TEXT_INPUT",
      "init_image_data": "$INIT_IMAGE_URL1",
      "init_image_strength": $INIT_IMAGE_STRENGTH2,
      "n_samples": $NSAMPLES,
      "seed": $SEED
    }
}
EOM
)

REQUEST13=$(cat <<- EOM
{
    "input": {
      "mode": "remix",
      "text_input": "",
      "init_image_data": "$INIT_IMAGE_URL1",
      "init_image_strength": $INIT_IMAGE_STRENGTH1,
      "seed": $SEED
    }
}
EOM
)

REQUEST14=$(cat <<- EOM
{
    "input": {
      "mode": "generate",
      "text_input": "$TEXT_INPUT",
      "steps": $STEPS,
      "width": $WIDTH,
      "height": $HEIGHT,
      "seed": $SEED
    }
}
EOM
)

REQUEST15=$(cat <<- EOM
{
    "input": {
      "mode": "remix",
      "text_input": "folded paper, origami",
      "init_image_data": "$INIT_IMAGE_URL1",
      "init_image_strength": $INIT_IMAGE_STRENGTH2,
      "upscale_f": $UPSCALE_F,
      "ip_image_strength": 0.5,
      "seed": $SEED
    }
}
EOM
)

REQUEST16=$(cat <<- EOM
{
    "input": {
      "mode": "remix",
      "text_input": "folded paper, origami",
      "init_image_data": "$INIT_IMAGE_URL1",
      "init_image_strength": $INIT_IMAGE_STRENGTH1,
      "lora": "$LORA_URL",
      "seed": $SEED
    }
}
EOM
)

REQUEST17=$(cat <<- EOM
{
    "input": {
      "mode": "controlnet",
      "text_input": "$TEXT_INPUT",
      "init_image_data": "$INIT_IMAGE_URL1",
      "init_image_strength": $INIT_IMAGE_STRENGTH_CONTROL2,
      "controlnet_type": "canny-edge",
      "seed": $SEED
    }
}
EOM
)

REQUEST18=$(cat <<- EOM
{
    "input": {
      "mode": "generate",
      "text_input": "$TEXT_INPUT",
      "steps": $STEPS,
      "width": $WIDTH,
      "height": $HEIGHT,
      "seed": $SEED
    }
}
EOM
)

REQUEST19=$(cat <<- EOM
{
    "input": {
      "mode": "controlnet",
      "text_input": "a photo of <concept>",
      "init_image_data": "$INIT_IMAGE_URL1",
      "init_image_strength": $INIT_IMAGE_STRENGTH_CONTROL1,
      "lora": "$LORA_URL",
      "controlnet_type": "luminance",
      "seed": $SEED
    }
}
EOM
)

REQUEST20=$(cat <<- EOM
{
    "input": {
      "mode": "blend",
      "interpolation_init_images": "$INIT_IMAGE_URL1|$INIT_IMAGE_URL2",
      "interpolation_init_images_min_strength": 0.05,
      "seed": $SEED
    }
}
EOM
)

REQUEST21=$(cat <<- EOM
{
    "input": {
      "mode": "upscale",
      "width": 1600,
      "height": 1600,
      "init_image_data": "$INIT_IMAGE_URL1",
      "init_image_strength": $INIT_IMAGE_STRENGTH_CONTROL1,
      "controlnet_type": "canny-edge",
      "seed": $SEED
    }
}
EOM
)

REQUEST22=$(cat <<- EOM
{
    "input": {
      "mode": "upscale",
      "width": 1400,
      "height": 1400,
      "init_image_data": "$INIT_IMAGE_URL1",
      "init_image_strength": $INIT_IMAGE_STRENGTH_CONTROL2,
      "seed": $SEED
    }
}
EOM
)


# Add all request variables to an array
REQUESTS=( "$REQUEST1" "$REQUEST2" "$REQUEST3" "$REQUEST4" "$REQUEST5" "$REQUEST6" "$REQUEST7" "$REQUEST8" "$REQUEST9" "$REQUEST10" "$REQUEST11" "$REQUEST12" "$REQUEST13" "$REQUEST14" "$REQUEST15" "$REQUEST16" "$REQUEST17" "$REQUEST18" "$REQUEST19" "$REQUEST20" "$REQUEST21" "$REQUEST22" )

# Randomly shuffle the array
RANDOMIZED_REQUESTS=( $(shuf -e "${REQUESTS[@]}") )

# Execute the requests in random order
for REQ in "${RANDOMIZED_REQUESTS[@]}"; do
  echo "Debug: The request being sent is: $REQ"
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" \
       -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" \
       -d "$REQ" | jq '.' >> output.log 2>> error.log
done