

# Define variables

TEXT_INPUT1="a photo of a magic stone portal in the forest"
TEXT_INPUT2="a photo of a massive banana on the summit of a mountain"

INIT_IMAGE_URL1="https://storage.googleapis.com/public-assets-xander/A_workbox/init_imgs/img_00003.jpg"
INIT_IMAGE_URL2="https://storage.googleapis.com/public-assets-xander/A_workbox/init_imgs/img_00005.jpg"
INIT_IMAGE_URL3="https://storage.googleapis.com/public-assets-xander/A_workbox/init_imgs/img_00006.jpg"

LORA_URL="https://edenartlab-prod-data.s3.us-east-1.amazonaws.com/92fb4b30d0d7f488998fc61f6ae8517dbf5f7c1d9f69fa4de71d25848987a21a.tar"

INIT_IMAGE_STRENGTH1=0.0
INIT_IMAGE_STRENGTH2=0.2
INIT_IMAGE_STRENGTH_CONTROL1=0.4
INIT_IMAGE_STRENGTH_CONTROL2=0.7

STEPS=25
WIDTH=768
HEIGHT=768
N_FRAMES=24

# STEPS=35
# WIDTH=1024
# HEIGHT=1024

UPSCALE_F=1.25
NSAMPLES=2
SEED=0


RUN_REAL2REAL=1
RUN_INTERPOLATE=1


###########################################################

# Delete output.log if it exists
[ -e "output.log" ] && rm output.log

# Delete error.log if it exists
[ -e "error.log" ] && rm error.log


################################################

if [ $RUN_INTERPOLATE -eq 1 ]; then
  echo "Running interpolate commands..."

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF | jq '.' >> output.log 2>> error.log
{
    "input": {
      "mode": "interpolate",
      "interpolation_texts": "$TEXT_INPUT1|$TEXT_INPUT2",
      "init_image": "$INIT_IMAGE_URL1",
      "init_image_strength": $INIT_IMAGE_STRENGTH1,
      "steps": $STEPS,
      "width": $WIDTH,
      "height": $HEIGHT,
      "n_frames": $N_FRAMES,
      "n_film": 1,
      "seed": $SEED
    }
}
EOF

else
  echo "Skipping interpolate commands"
fi


########## REAL2REAL COMMANDS: ##########

if [ $RUN_REAL2REAL -eq 1 ]; then
  echo "Running real2real commands..."

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF | jq '.' >> output.log 2>> error.log
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
EOF

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF | jq '.' >> output.log 2>> error.log
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
EOF

else
  echo "Skipping real2real commands"
fi

###############################################################################
  
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF | jq '.' >> output.log 2>> error.log
{
    "input": {
      "mode": "interpolate",
      "interpolation_texts": "$TEXT_INPUT1|$TEXT_INPUT2",
      "init_image": "$INIT_IMAGE_URL1",
      "init_image_strength": $INIT_IMAGE_STRENGTH1,
      "steps": $STEPS,
      "width": $WIDTH,
      "height": $HEIGHT,
      "n_frames": $N_FRAMES,
      "n_film": 1,
      "seed": $SEED
    }
}
EOF

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF | jq '.' >> output.log 2>> error.log
{
    "input": {
      "mode": "interpolate",
      "interpolation_texts": "$TEXT_INPUT1|$TEXT_INPUT2",
      "control_image": "$INIT_IMAGE_URL1",
      "control_image_strength": $INIT_IMAGE_STRENGTH2,
      "controlnet_type": "luminance",
      "steps": $STEPS,
      "width": $WIDTH,
      "height": $HEIGHT,
      "n_frames": $N_FRAMES,
      "n_film": 1,
      "seed": $SEED
    }
}
EOF

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF | jq '.' >> output.log 2>> error.log
{
    "input": {
      "mode": "interpolate",
      "interpolation_texts": "$TEXT_INPUT1|$TEXT_INPUT2",
      "control_image": "$INIT_IMAGE_URL1",
      "control_image_strength": $INIT_IMAGE_STRENGTH2,
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
EOF



########## very HD REAL2REAL: ############
##########################################