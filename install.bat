@echo off
setlocal

rem Step 1: Check if Python 3.10 is installed
python --version | findstr "3.10"
if %errorlevel% neq 0 (
    echo Python 3.10 is required. Please install it and run this script again.
    exit /b 1
)

rem Step 2: Create a virtual environment
python -m venv venv

rem Step 3: Activate the virtual environment
call venv\Scripts\activate

rem Step 4: Install dependencies
pip3 install clean-fid numba numpy "torch==2.0.0+cu118" "torchvision==0.15.1+cu118" "tsp_solver" "ftfy==6.1.1" "scipy==1.11.1" "transformers==4.30.2" "accelerate==0.20.3" "mediapipe==0.9.1.0" "matplotlib" "scikit-image==0.19.3" "tokenizers==0.13.3" "triton==2.0.0" "apache-beam==2.45.0" "natsort==8.2.0" "mediapy==1.1.4" "wandb==0.13.10" "invisible-watermark>=0.2.0" "timm==0.9.6" "einops" "opencv-python" "moviepy" "pandas" "lpips" "open-clip-torch" "omegaconf"

rem Step 5: Perform pre-install steps
git clone -b sdxl https://www.github.com/abraham-ai/diffusers
pip install -e ./diffusers

git clone https://github.com/pharmapsychotic/BLIP
pip install -e ./BLIP

git clone https://github.com/pharmapsychotic/clip-interrogator
pip install -e ./clip-interrogator --no-deps

pip install tensorflow==2.11.0

git clone https://github.com/abraham-ai/frame-interpolation

rem Step 6: Check if all required libraries are installed
pip list | findstr /i "einops opencv-python moviepy pandas lpips open-clip-torch omegaconf" >nul
if %errorlevel% neq 0 (
    echo Some required libraries are missing. Attempting to install them...
    pip install einops opencv-python moviepy pandas lpips open-clip-torch omegaconf
)

rem Step 7: Create folders and download files using PowerShell Invoke-WebRequest if they don't exist
mkdir models\checkpoints\sdxl-v1.0
cd models\checkpoints\sdxl-v1.0

if not exist sd_xl_base_1.0_0.9vae.safetensors (
    powershell -command "& { Invoke-WebRequest -Uri 'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors' -OutFile 'sd_xl_base_1.0_0.9vae.safetensors' }"
)

if not exist model_index.json (
    powershell -command "& { Invoke-WebRequest -Uri 'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/model_index.json' -OutFile 'model_index.json' }"
)

rem add ip_adapter dependencies

cd ..
cd ..
mkdir G:\AI\eden-sd-pipelines\models\ip_adapter\image_encoder

rem finish adding models here

rem Step 8: Deactivate the virtual environment
cd ../../../..
deactivate

echo Virtual environment setup is complete. You can activate it by running "call venv\Scripts\activate".
