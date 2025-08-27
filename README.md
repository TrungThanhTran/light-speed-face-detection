# a super light speed face detection
Yunet is one of the fastest (maybe lightest) face detection in the world with only 75M params but achieve pretty well performance. There are some thing we can improve on top of it and this repo is to do so.

This repo strongly based on [Yunet training repo](https://github.com/ShiqiYu/libfacedetection.train).


# Updated: 
   Since 3 years from the lastest update for Yunet tranining repo, in order to install necessary libs please use the following sets:
   
   ### Fresh venv recommended
   1. Install torch <= I'm using Cuda 12.9 - NVIDIA 3090
  ```shell
   pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 \
   -f https://download.pytorch.org/whl/torch_stable.html
  ```
  2. Install mmcv
  ```shell
   pip install -U openmim
   mim install "mmcv-full==1.6.0" \
   -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12/index.html
  ```
