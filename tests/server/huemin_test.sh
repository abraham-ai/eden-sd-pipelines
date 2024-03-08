

# Define variables

TEXT_INPUT="a photo of a magic stone portal in the forest"
INIT_IMAGE_URL1="https://storage.googleapis.com/public-assets-xander/A_workbox/init_imgs/img_00003.jpg"
INIT_IMAGE_URL2="https://storage.googleapis.com/public-assets-xander/A_workbox/init_imgs/img_00005.jpg"
INIT_IMAGE_URL3="https://storage.googleapis.com/public-assets-xander/A_workbox/init_imgs/img_00006.jpg"

LORA_FACE_URL="https://edenartlab-prod-data.s3.us-east-1.amazonaws.com/74dd58b51abb1407c66f47f39031ab65a3bd516ba3604fbc2dc4709d64faf325.tar"
LORA_STYLE_URL="https://edenartlab-prod-data.s3.us-east-1.amazonaws.com/ac64de34e28cb7cb3f6cd53f3a408660b9f0f805b6f0a7555cf601ba482dd75b.tar"
LORA_OBJECT_URL="https://edenartlab-prod-data.s3.us-east-1.amazonaws.com/cb34799e2b04671cfd4c78104c0eed4e68866e5adbf9f21d40bf2ca8d4db08a7.tar"

INIT_IMAGE_STRENGTH1=0.0
INIT_IMAGE_STRENGTH2=0.2
INIT_IMAGE_STRENGTH_CONTROL1=0.4
INIT_IMAGE_STRENGTH_CONTROL2=0.7

STEPS=20
WIDTH=512
HEIGHT=512

# STEPS=35
# WIDTH=1024
# HEIGHT=1024

UPSCALE_F=1.1
NSAMPLES=2
SEED=0


RUN_CREATE=1
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

echo "Running create commands..."

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF

  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF
  curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H 'Content-Type: application/json' "http://0.0.0.0:5000/predictions" -d @- <<EOF
{
    "input": {
      "mode": "kojii/huemin",
      "text_input": "isometric generative landscape orthographic abstract aj casson perlin noise 3d shaders areal embroidery minimalism claude monet oil painting pastel"
    }
}
EOF

