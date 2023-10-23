

# Define variables

TEXT_INPUT="a photo of a magic stone portal in the forest"
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

# STEPS=35
# WIDTH=1024
# HEIGHT=1024

UPSCALE_F=1.25
NSAMPLES=2
SEED=0


RUN_GENERATE=1
RUN_REMIX=1
RUN_CONTROLNET=1
RUN_BLEND=1
RUN_UPSCALE=1


###########################################################

# Delete output.log if it exists
[ -e "output.log" ] && rm output.log

# Delete error.log if it exists
[ -e "error.log" ] && rm error.log


####### GENEREATE COMMANDS: #######

if [ $RUN_GENERATE -eq 1 ]; then
  echo "Running generate commands..."

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
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
EOF

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "generate",
      "text_input": "a photo of <concept> on the beach, drinking Coca-Cola",
      "lora": "$LORA_URL",
      "seed": $SEED
    }
}
EOF

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
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
EOF

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
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
EOF

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF | jq '.' >> output.log 2>> error.log
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
EOF

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF | jq '.' >> output.log 2>> error.log
{
    "input": {
      "mode": "generate",
      "text_input": "$TEXT_INPUT",
      "init_image": "$INIT_IMAGE_URL1",
      "init_image_strength": $INIT_IMAGE_STRENGTH1,
      "seed": $SEED
    }
}
EOF

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF | jq '.' >> output.log 2>> error.log
{
    "input": {
      "mode": "generate",
      "text_input": "$TEXT_INPUT",
      "init_image": "$INIT_IMAGE_URL1",
      "init_image_strength": $INIT_IMAGE_STRENGTH2,
      "n_samples": $NSAMPLES,
      "seed": $SEED
    }
}
EOF

else
  echo "Skipping generate commands"
fi

###############################################################################


if [ $RUN_REMIX -eq 1 ]; then
  echo "Running remix commands..."

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF | jq '.' >> output.log 2>> error.log
{
    "input": {
      "mode": "remix",
      "text_input": "",
      "init_image": "$INIT_IMAGE_URL1",
      "init_image_strength": $INIT_IMAGE_STRENGTH1,
      "seed": $SEED
    }
}
EOF

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
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
EOF

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF | jq '.' >> output.log 2>> error.log
{
    "input": {
      "mode": "remix",
      "text_input": "folded paper, origami",
      "init_image": "$INIT_IMAGE_URL1",
      "init_image_strength": $INIT_IMAGE_STRENGTH2,
      "upscale_f": $UPSCALE_F,
      "ip_image_strength": 0.5,
      "seed": $SEED
    }
}
EOF

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF | jq '.' >> output.log 2>> error.log
{
    "input": {
      "mode": "remix",
      "text_input": "folded paper, origami",
      "init_image": "$INIT_IMAGE_URL1",
      "init_image_strength": $INIT_IMAGE_STRENGTH1,
      "lora": "$LORA_URL",
      "seed": $SEED
    }
}
EOF

else
  echo "Skipping remix commands"
fi


############################# CONTROLNET: ##################################################


if [ $RUN_CONTROLNET -eq 1 ]; then
  echo "Running controlnet commands..."

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF | jq '.' >> output.log 2>> error.log
{
    "input": {
      "mode": "controlnet",
      "text_input": "$TEXT_INPUT",
      "control_image": "$INIT_IMAGE_URL1",
      "control_image_strength": $INIT_IMAGE_STRENGTH_CONTROL2,
      "controlnet_type": "canny-edge",
      "seed": $SEED
    }
}
EOF

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
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
EOF

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF | jq '.' >> output.log 2>> error.log
{
    "input": {
      "mode": "controlnet",
      "text_input": "a photo of <concept>",
      "control_image": "$INIT_IMAGE_URL1",
      "control_image_strength": $INIT_IMAGE_STRENGTH_CONTROL1,
      "lora": "$LORA_URL",
      "controlnet_type": "luminance",
      "seed": $SEED
    }
}
EOF


else
  echo "Skipping controlnet commands"
fi



############################# BLEND: ##################################################



if [ $RUN_BLEND -eq 1 ]; then
  echo "Running blend commands..."

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF | jq '.' >> output.log 2>> error.log
{
    "input": {
      "mode": "blend",
      "interpolation_init_images": "$INIT_IMAGE_URL1|$INIT_IMAGE_URL2",
      "interpolation_init_images_min_strength": 0.05,
      "seed": $SEED
    }
}
EOF

else
  echo "Skipping blend commands"
fi


############################# UPSCALE: ################################################

if [ $RUN_UPSCALE -eq 1 ]; then
  echo "Running UPSCALE commands..."

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF | jq '.' >> output.log 2>> error.log
{
    "input": {
      "mode": "upscale",
      "width": 1600,
      "height": 1600,
      "control_image": "$INIT_IMAGE_URL1",
      "control_image_strength": $INIT_IMAGE_STRENGTH_CONTROL1,
      "controlnet_type": "canny-edge",
      "seed": $SEED
    }
}
EOF

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF | jq '.' >> output.log 2>> error.log
{
    "input": {
      "mode": "upscale",
      "width": 1400,
      "height": 1400,
      "init_image": "$INIT_IMAGE_URL1",
      "init_image_strength": $INIT_IMAGE_STRENGTH_CONTROL2,
      "seed": $SEED
    }
}
EOF

else
  echo "Skipping upscale commands"
fi
