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

   ### Improvement
   1. Use takenique SE - squeeze and excitation block
   2. Increase the size of model from 'nano' to 'tiny', it's able to increase to medium or large if we need but trade-off for x% speed. But the speed is already high
   3. Increase batch size training from 16 to 32 or higher also contribute to the mAP
   4. I also add other dataset like: face in COCO2017, Face Detection Kaggle, Lagenda but it seems no improvement due to the different in data distribution (Maybe)


   ### Training
   ```shell
   CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh ./configs/yunet_n.py 2 12345
   ```

   ### Testing on Wider face
   ```shell
   python tools/test_widerface.py ./configs/yunet_n.py ./weights/yunet_n.pth --mode 2
   ```

   ### Result
   ```shell
   Baseline Yunet_n: AP_easy=0.892, AP_medium=0.883, AP_hard=0.811
   SE Yunet_n: AP_easy=0.892, AP_medium=0.884, AP_hard=0.815 only + about 10% params
   Tiny Yunet_t: AP_easy=0.906, AP_medium=0.896, AP_hard=0.829 + about 30% params
   ```
