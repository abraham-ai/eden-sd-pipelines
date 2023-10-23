

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

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF | jq '.' >> output.log 2>> error.log
{
    "input": {
      "mode": "upscale",
      "width": 1600,
      "height": 1600,
      "init_image": "$INIT_IMAGE_URL1",
      "init_image_strength": 0.2,
      "control_image": "$INIT_IMAGE_URL1",
      "control_image_strength": 0.7,
      "controlnet_type": "canny-edge",
      "seed": $SEED
    }
}
EOF