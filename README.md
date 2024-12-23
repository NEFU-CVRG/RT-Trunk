# RT-Trunk: Real-time instance segmentation of tree trunks from under-canopy images in complex forest environments
- This is an official implementations for "Real-time instance segmentation of tree trunks from under-canopy images in complex forest environments"
- RT-Trunk takes SparseInst as the basic framework, uses ConvNeXt as the backbone and introduces the CBAM (Convolutional Block Attention Module) attention mechanism.
- The implementation of RT-Trunk is based on [mmdetection](https://github.com/open-mmlab/mmdetection), [mmpretrain](https://github.com/open-mmlab/mmpretrain) and [SparseInst](https://github.com/hustvl/SparseInst#models).
# Install
- Install mmdetection according to the [official tutorial](https://mmdetection.readthedocs.io/en/latest/get_started.html).
- Install mmpretrain according to the [official tutorial](https://mmpretrain.readthedocs.io/en/latest/get_started.html#installation).
- Enter the "projects" folder under the mmdetection directory, and then execute the following commands:
  ```git
  git clone https://github.com/NEFU-CVRG/RT-Trunk.git
- Download the [pre-trained weight](https://github.com/open-mmlab/mmpretrain/tree/main/configs/convnext) of ConvNeXt from mmpretrain and put it into the pretrained_weights folder under the RT-Trunk directory.
- Download the pre-trained weight of SparseInst from the [official model zoo](https://github.com/hustvl/SparseInst#models) of SparseInst and put it into the pretrained_weights folder under the RT-Trunk directory.
# Dataset Preparation
- Prepare your dataset according to the COCO data format and with reference to the [dataset preparation tutorial](https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html) of mmdetection.
# Training
```
python tools/train.py projects/RT-Trunk/configs/RTtrunk_1x-ms-270k_coco.py
```
# Acknowledgement
We sincerely thank [mmdetection](https://github.com/open-mmlab/mmdetection), [mmpretrain](https://github.com/open-mmlab/mmpretrain), [SparseInst](https://github.com/hustvl/SparseInst), [yolov4-tiny-pytorch](https://github.com/bubbliiiing/yolov4-tiny-pytorch) for providing their wonderful code to the community!
