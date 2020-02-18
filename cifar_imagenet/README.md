# CIFAR & IMAGENET

## Requirements
This code is tested inside the NVIDIA Pytorch docker container release 19.09. This container can be pulled from NVIDIA GPU Cloud as follows:

`docker pull nvcr.io/nvidia/pytorch:19.09-py3`

Detailed information on packages included in the NVIDIA Pytorch containter 19.09 can be found at [NVIDIA Pytorch Release 19.09](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_19-09.html#rel_19-09). In addition to those packages, the following packages are required:

- Sklearn: `pip install -U scikit-learn --user`
- OpenCV: `pip install opencv-python`
- Progress: `pip install progress`

In order to reproduce the plots in our papers, the following packages are needed:

- Pandas: `pip install pandas`
- Seaborn: `pip install seaborn`

To run our code without using the NVIDIA Pytorch containter, at least the following packages are required:

- Ubuntu 18.04 including Python 3.6 environment
- PyTorch 1.2.0
- NVIDIA CUDA 10.1.243 including cuBLAS 10.2.1.243
- NVIDIA cuDNN 7.6.3
- [NVIDIA APEX](https://github.com/NVIDIA/apex)

## ImageNet Experiments Requires ImageNet Datasets in LMDB Format
Using the dafault `datasets.ImageFolder` + `data.DataLoader` is not efficient due to the slow reading of discontinuous small chunks. In order to speed up the training on ImageNet, we convert small JPEG images into a large binary file in Lighting Memory-Mapped Database (LMDB) format and load the training data with `data.distributed.DistributedSampler` and `data.DataLoader`. You can follow the [instructions](http://caffe.berkeleyvision.org/gathered/examples/imagenet.html) for Caffe to build the LMDB dataset of ImageNet. Alternatively, you can use these following two sets of instructions to build the LMDB dataset of ImageNet:[https://github.com/intel/caffe/wiki/How-to-create-ImageNet-LMDB](https://github.com/intel/caffe/wiki/How-to-create-ImageNet-LMDB) and [https://github.com/rioyokotalab/caffe/wiki/How-to-Create-Imagenet-ILSVRC2012-LMDB](https://github.com/rioyokotalab/caffe/wiki/How-to-Create-Imagenet-ILSVRC2012-LMDB).

The ImageNet LMDB dataset should be placed inside the directory `/datasets/imagenet` in your computer and contains the following files:

`fid_mean_cov.npz`  `train_faster_imagefolder.lmdb`  `train_faster_imagefolder.lmdb.pt`  `val_faster_imagefolder.lmdb`  `val_faster_imagefolder.lmdb.pt`

## Code for Plotting Figures in Our Paper
We provide code for plotting figures in our paper in the jupyter notebook `plot_code_srsgd.ipynb`. For Figure 6 in the Appendix, we follow this github: `https://github.com/wronnyhuang/gen-viz/tree/master/minefield`. Instead of using SGD, we trained the model using SRSGD and plotted the trajectories. Since this visualization code took 2 or 3 days to finish, we didn't include it in `plot_code_srsgd.ipynb`. 

## Training
A training [recipe](/cifar_imagenet/recipes.md) is provided for image classification experiments. The recipe contains the commands to run experiments for Table 1, 2, 3, 4 and 5 in our paper, which includes full and short trainings on CIFAR10, CIFAR100, and ImageNet using SRSGD, as well as the baseline SGD trainings. Other experiments in our paper and the appendix can be run using the same `cifar.py` and `imagenet.py` files with the different values of parameters.

