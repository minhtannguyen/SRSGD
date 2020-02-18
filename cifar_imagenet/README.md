# CIFAR & IMAGENET

## Requirements
This code is tested inside the NVIDIA Pytorch docker container release 19.09. This container can be pulled from NVIDIA GPU Cloud as follows:

`docker pull nvcr.io/nvidia/pytorch:19.09-py3`

Detailed information on packages included in the NVIDIA Pytorch containter 19.09 can be found at [NVIDIA Pytorch Release 19.09](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_19-09.html#rel_19-09). In addition to packages included in the NVIDIA Pytorch containter 19.09 , the following packages are required:

- Sklearn: `pip install -U scikit-learn --user`
- OpenCV: `pip install opencv-python`
- Progress: `pip install progress`

In order to reproduce the plots in our papers, the following packages are needed:

- Pandas: `pip install pandas`
- Seaborn: `pip install seaborn`

To run our code without using the NVIDIA Pytorch containters, at least the following packages are required:

- Ubuntu 18.04 including Python 3.6 environment
- PyTorch 1.2.0
- NVIDIA CUDA 10.1.243 including cuBLAS 10.2.1.243
- NVIDIA cuDNN 7.6.3
- NVIDIA APEX

## Training
A training [recipe](/cifar_imagenet/recipes.md) is provided for image classification experiments. 

