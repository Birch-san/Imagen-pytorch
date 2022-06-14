# Setup repository

I've shared these instructions in a gist too:  
https://gist.github.com/Birch-san/e1861d3af0f262dfb737ebb650eb8c4e

```bash
# brew install git-lfs openblas
brew install git-lfs
git lfs install
git clone https://huggingface.co/Cene655/ImagenT5-3B
git clone https://github.com/xinntao/Real-ESRGAN.git
python3 -m venv venv
source ./venv/bin/activate
python3 -m pip install --upgrade pip
pip install wheel
pip install .
pip install git+https://github.com/openai/CLIP.git
pip install basicsr
pip install facexlib
# if numpy tries to build-from-source, it will look for llvm-ar which doesn't exist in XCode toolchain
# AR='/usr/bin/ar' OPENBLAS="$(brew --prefix openblas)" pip install gfpgan
# Nix version:
# AR='/usr/bin/ar' OPENBLAS="$HOME/.nix-profile/lib/libopenblas.dylib" pip install gfpgan
pip install git+https://github.com/Birch-san/GFPGAN.git@newer-numpy
cd Real-ESRGAN
pip install -r requirements.txt
python setup.py develop
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models
cd ..
pip install ipython
# our torch nightly probably got nuked by the above, but we do need it for GPU support on macOS
pip install --pre "torch>1.12.0.dev0,<1.13.0.dev20220611" "torchvision>0.14.0.dev0" "torchaudio>0.12.0.dev0" --extra-index-url https://download.pytorch.org/whl/nightly/cpu
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth
```