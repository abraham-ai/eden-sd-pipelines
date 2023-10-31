

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
RUN_REAL2REAL=1

if [ $RUN_REAL2REAL -eq 1 ]; then
  echo "Running real2real commands..."

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF | jq '.' >> output.log 2>> error.log
{
    "input": {
      "mode": "real2real",
      "interpolation_init_images": "$INIT_IMAGE_URL1|$INIT_IMAGE_URL2",
      "lora": "$LORA_URL",
      "steps": $STEPS,
      "width": 512,
      "height": 640,
      "n_frames": 20,
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
      "width": 500,
      "height": 500,
      "n_frames": 120,
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
      "width": 640,
      "height": 640,
      "n_frames": 130,
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
      "width": 640,
      "height": 500,
      "n_frames": 60,
      "seed": $SEED
    }
}
EOF

else
  echo "Skipping real2real commands"
fi
