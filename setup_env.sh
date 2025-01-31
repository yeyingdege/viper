conda create -n vipergpt python=3.10
conda activate vipergpt
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install --upgrade bitsandbytes
