mkdir /cog/
cd /cog/
git clone https://github.com/abraham-ai/eden-sd-pipelines

curl https://sh.rustup.rs -sSf | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyyaml

python -m pip install transformers==4.26.1 tokenizers==0.13.2
python -m pip install clip-interrogator==0.5.4
python -m pip install pyre-extensions==0.0.23
python -m pip install xformers==0.0.17.dev442
python -m pip install tsp_solver2 moviepy einops safetensors opencv-python fire lpips pandas accelerate omegaconf

git clone https://github.com/abraham-ai/diffusers
cd diffusers
python -m pip install -e .

cd /cog/eden-sd-pipelines/eden/templates
python unit_test_all_templates
