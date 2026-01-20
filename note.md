watch -n 1 nvidia-smi

pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130


torch==2.9.0.*
torchvision==0.24.0.*
torchaudio==2.9.0.*
opencv-python==4.12.0.88
Pillow==12.0.0
tqdm==4.67.1


tensorboard
tensorboard --logdir=detection/runs
    