
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


# Add all request variables to an array
REQUESTS=( "$REQUEST1" "$REQUEST2" "$REQUEST3" "$REQUEST4" )

# Validate each request JSON
for REQ in "${REQUESTS[@]}"; do
  echo $REQ | jq empty
  if [ $? -ne 0 ]; then
    echo "Invalid JSON found: $REQ"
    exit 1
  fi
done

# Create a single string with all requests separated by a unique delimiter
ALL_REQUESTS=$(printf "@@@%s" "${REQUESTS[@]}")

# Set IFS to the unique delimiter and read into an array
IFS="@@@" read -ra RANDOMIZED_REQUESTS <<< "$ALL_REQUESTS"

# Now, RANDOMIZED_REQUESTS should be properly populated. You can shuffle if needed.
RANDOMIZED_REQUESTS=( $(shuf -e "${RANDOMIZED_REQUESTS[@]}") )

# Execute the requests
for REQ in "${RANDOMIZED_REQUESTS[@]}"; do
  if [ ! -z "$REQ" ]; then  # Skip empty strings
    echo "Debug: The request being sent is: $REQ"
    RESPONSE=$(curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" \
                    -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" \
                    -d "$REQ")
    echo "Debug: The response is: $RESPONSE"
    echo $RESPONSE | jq '.' >> output.log 2>> error.log
  fi
done