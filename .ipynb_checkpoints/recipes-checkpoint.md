# SRSGD for CIFAR10 using linear schedule

```
CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar10" --depth 110 --epochs 200 --schedule 80 120 160 --restart-schedule 30 60 90 120 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar10-preresnet110-srsgd-linear-schedule --gpu-id 0 --model_name preresnet110_srsgd --optimizer "srsgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar10" --depth 290 --epochs 200 --schedule 80 120 160 --restart-schedule 30 60 90 120 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar10-preresnet290-srsgd-linear-schedule --gpu-id 0 --model_name preresnet290_srsgd --optimizer "srsgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar10" --depth 470 --epochs 200 --schedule 80 120 160 --restart-schedule 30 60 90 120 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar10-preresnet470-srsgd-linear-schedule --gpu-id 0 --model_name preresnet470_srsgd --optimizer "srsgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar10" --depth 650 --epochs 200 --schedule 80 120 160 --restart-schedule 30 60 90 120 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar10-preresnet650-srsgd-linear-schedule --gpu-id 0 --model_name preresnet650_srsgd --optimizer "srsgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0,1,2,3 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar10" --depth 1001 --epochs 200 --schedule 80 120 160 --restart-schedule 30 60 90 120 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar10-preresnet1001-srsgd-linear-schedule --gpu-id "0,1,2,3" --model_name preresnet1001_srsgd --optimizer "srsgd" --manualSeed 0
```

# SRSGD for CIFAR10 using exponential schedule

```
CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar10" --depth 110 --epochs 200 --schedule 80 120 160 --restart-schedule 40 50 63 78 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar10-preresnet110-srsgd-exponential-schedule --gpu-id 0 --model_name preresnet110_srsgd --optimizer "srsgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar10" --depth 290 --epochs 200 --schedule 80 120 160 --restart-schedule 40 50 63 78 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar10-preresnet290-srsgd-exponential-schedule --gpu-id 0 --model_name preresnet290_srsgd --optimizer "srsgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar10" --depth 470 --epochs 200 --schedule 80 120 160 --restart-schedule 40 50 63 78 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar10-preresnet470-srsgd-exponential-schedule --gpu-id 0 --model_name preresnet470_srsgd --optimizer "srsgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar10" --depth 650 --epochs 200 --schedule 80 120 160 --restart-schedule 40 50 63 78 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar10-preresnet650-srsgd-exponential-schedule --gpu-id 0 --model_name preresnet650_srsgd --optimizer "srsgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0,1,2,3 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar10" --depth 1001 --epochs 200 --schedule 80 120 160 --restart-schedule 40 50 63 78 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar10-preresnet1001-srsgd-exponential-schedule --gpu-id "0,1,2,3" --model_name preresnet1001_srsgd --optimizer "srsgd" --manualSeed 0
```

# SRSGD for CIFAR100 using linear schedule

```
CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar100" --depth 110 --epochs 200 --schedule 80 120 160 --restart-schedule 50 100 150 200 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar100-preresnet110-srsgd-linear-schedule --gpu-id 0 --model_name preresnet110_srsgd --optimizer "srsgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar100" --depth 290 --epochs 200 --schedule 80 120 160 --restart-schedule 50 100 150 200 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar100-preresnet290-srsgd-linear-schedule --gpu-id 0 --model_name preresnet290_srsgd --optimizer "srsgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar100" --depth 470 --epochs 200 --schedule 80 120 160 --restart-schedule 50 100 150 200 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar100-preresnet470-srsgd-linear-schedule --gpu-id 0 --model_name preresnet470_srsgd --optimizer "srsgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar100" --depth 650 --epochs 200 --schedule 80 120 160 --restart-schedule 50 100 150 200 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar100-preresnet650-srsgd-linear-schedule --gpu-id 0 --model_name preresnet650_srsgd --optimizer "srsgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0,1,2,3 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar100" --depth 1001 --epochs 200 --schedule 80 120 160 --restart-schedule 50 100 150 200 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar100-preresnet1001-srsgd-linear-schedule --gpu-id "0,1,2,3" --model_name preresnet1001_srsgd --optimizer "srsgd" --manualSeed 0
```

# SRSGD for CIFAR100 using exponential schedule

```
CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar100" --depth 110 --epochs 200 --schedule 80 120 160 --restart-schedule 45 68 101 152 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar100-preresnet110-srsgd-exponential-schedule --gpu-id 0 --model_name preresnet110_srsgd --optimizer "srsgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar100" --depth 290 --epochs 200 --schedule 80 120 160 --restart-schedule 45 68 101 152 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar100-preresnet290-srsgd-exponential-schedule --gpu-id 0 --model_name preresnet290_srsgd --optimizer "srsgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar100" --depth 470 --epochs 200 --schedule 80 120 160 --restart-schedule 45 68 101 152 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar100-preresnet470-srsgd-exponential-schedule --gpu-id 0 --model_name preresnet470_srsgd --optimizer "srsgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar100" --depth 650 --epochs 200 --schedule 80 120 160 --restart-schedule 45 68 101 152 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar100-preresnet650-srsgd-exponential-schedule --gpu-id 0 --model_name preresnet650_srsgd --optimizer "srsgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0,1,2,3 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar100" --depth 1001 --epochs 200 --schedule 80 120 160 --restart-schedule 45 68 101 152 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar100-preresnet1001-srsgd-exponential-schedule --gpu-id "0,1,2,3" --model_name preresnet1001_srsgd --optimizer "srsgd" --manualSeed 0
```

# SGD for CIFAR10

```
CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar10" --depth 110 --epochs 200 --schedule 80 120 160 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar10-preresnet110-sgd --gpu-id 0 --model_name preresnet110_sgd --optimizer "sgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar10" --depth 290 --epochs 200 --schedule 80 120 160 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar10-preresnet290-sgd --gpu-id 0 --model_name preresnet290_sgd --optimizer "sgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar10" --depth 470 --epochs 200 --schedule 80 120 160 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar10-preresnet470-sgd --gpu-id 0 --model_name preresnet470_sgd --optimizer "sgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar10" --depth 650 --epochs 200 --schedule 80 120 160 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar10-preresnet650-sgd --gpu-id 0 --model_name preresnet650_sgd --optimizer "sgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0,1,2,3 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar10" --depth 1001 --epochs 200 --schedule 80 120 160 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar10-preresnet1001-sgd --gpu-id "0,1,2,3" --model_name preresnet1001_sgd --optimizer "sgd" --manualSeed 0
```

# SGD for CIFAR100

```
CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar100" --depth 110 --epochs 200 --schedule 80 120 160 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar100-preresnet110-sgd --gpu-id 0 --model_name preresnet110_sgd --optimizer "sgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar100" --depth 290 --epochs 200 --schedule 80 120 160 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar100-preresnet290-sgd --gpu-id 0 --model_name preresnet290_sgd --optimizer "sgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar100" --depth 470 --epochs 200 --schedule 80 120 160 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar100-preresnet470-sgd --gpu-id 0 --model_name preresnet470_sgd --optimizer "sgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar100" --depth 650 --epochs 200 --schedule 80 120 160 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar100-preresnet650-sgd --gpu-id 0 --model_name preresnet650_sgd --optimizer "sgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0,1,2,3 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar100" --depth 1001 --epochs 200 --schedule 80 120 160 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar100-preresnet1001-sgd --gpu-id "0,1,2,3" --model_name preresnet1001_sgd --optimizer "sgd" --manualSeed 0
```

# SRSGD for CIFAR10 using short training and linear schedule

```
CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar10" --depth 110 --epochs 100 --schedule 80 90 95 --restart-schedule 30 60 90 120 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar10-preresnet110-srsgd-short-training-linear-schedule --gpu-id 0 --model_name preresnet110_srsgd --optimizer "srsgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar10" --depth 290 --epochs 100 --schedule 80 90 95 --restart-schedule 30 60 90 120 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar10-preresnet290-srsgd-short-training-linear-schedule --gpu-id 0 --model_name preresnet290_srsgd --optimizer "srsgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar10" --depth 470 --epochs 100 --schedule 80 90 95 --restart-schedule 30 60 90 120 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar10-preresnet470-srsgd-short-training-linear-schedule --gpu-id 0 --model_name preresnet470_srsgd --optimizer "srsgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar10" --depth 650 --epochs 100 --schedule 80 90 95 --restart-schedule 30 60 90 120 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar10-preresnet650-srsgd-short-training-linear-schedule --gpu-id 0 --model_name preresnet650_srsgd --optimizer "srsgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0,1,2,3 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar10" --depth 1001 --epochs 100 --schedule 80 90 95 --restart-schedule 30 60 90 120 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar10-preresnet1001-srsgd-short-training-linear-schedule --gpu-id "0,1,2,3" --model_name preresnet1001_srsgd --optimizer "srsgd" --manualSeed 0
```

# SRSGD for CIFAR100 using short training and linear schedule

```
CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar100" --depth 110 --epochs 100 --schedule 80 90 95 --restart-schedule 50 100 150 200 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar100-preresnet110-srsgd-short-training-linear-schedule --gpu-id 0 --model_name preresnet110_srsgd --optimizer "srsgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar100" --depth 290 --epochs 100 --schedule 80 90 95 --restart-schedule 50 100 150 200 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar100-preresnet290-srsgd-short-training-linear-schedule --gpu-id 0 --model_name preresnet290_srsgd --optimizer "srsgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar100" --depth 470 --epochs 100 --schedule 80 90 95 --restart-schedule 50 100 150 200 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar100-preresnet470-srsgd-short-training-linear-schedule --gpu-id 0 --model_name preresnet470_srsgd --optimizer "srsgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar100" --depth 650 --epochs 100 --schedule 80 90 95 --restart-schedule 50 100 150 200 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar100-preresnet650-srsgd-short-training-linear-schedule --gpu-id 0 --model_name preresnet650_srsgd --optimizer "srsgd" --manualSeed 0

CUDA_VISIBLE_DEVICES=0,1,2,3 python cifar.py -a "preresnet" --block-name "bottleneck" --dataset "cifar100" --depth 1001 --epochs 100 --schedule 80 90 95 --restart-schedule 50 100 150 200 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint ./experiments-restarting/cifar100-preresnet1001-srsgd-short-training-linear-schedule --gpu-id "0,1,2,3" --model_name preresnet1001_srsgd --optimizer "srsgd" --manualSeed 0
```

# SRSGD for ImageNet

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1 imagenet.py -j 16 -a "resnet50" --data /datasets/imagenet --epochs 90 --schedule 31 61 --restart-schedule 40 80 80 --gamma 0.1 --lr 0.1 --train-batch 32 --test-batch 25 -c ./experiments-restarting/imagenet-resnet50-srsgd --model_name resnet50_srsgd --optimizer "srsgd" --gpu-id "0,1,2,3,4,5,6,7" --manualSeed 0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1 imagenet.py -j 16 -a "resnet101" --data /datasets/imagenet --epochs 90 --schedule 31 61 --restart-schedule 40 80 80 --gamma 0.1 --lr 0.1 --train-batch 32 --test-batch 25 -c ./experiments-restarting/imagenet-resnet101-srsgd --model_name resnet101_srsgd --optimizer "srsgd" --gpu-id "0,1,2,3,4,5,6,7" --manualSeed 0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1 imagenet.py -j 16 -a "resnet152" --data /datasets/imagenet --epochs 90 --schedule 31 61 --restart-schedule 40 80 80 --gamma 0.1 --lr 0.1 --train-batch 32 --test-batch 25 -c ./experiments-restarting/imagenet-resnet152-srsgd --model_name resnet152_srsgd --optimizer "srsgd" --gpu-id "0,1,2,3,4,5,6,7" --manualSeed 0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1 imagenet.py -j 16 -a "resnet200" --data /datasets/imagenet --epochs 90 --schedule 31 61 --restart-schedule 40 80 80 --gamma 0.1 --lr 0.1 --train-batch 32 --test-batch 25 -c ./experiments-restarting/imagenet-resnet200-srsgd --model_name resnet200_srsgd --optimizer "srsgd" --gpu-id "0,1,2,3,4,5,6,7" --manualSeed 0
```

# SGD for ImageNet

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1 imagenet.py -j 16 -a "resnet50" --data /datasets/imagenet --epochs 90 --schedule 31 61 --gamma 0.1 --lr 0.1 --train-batch 32 --test-batch 25 -c ./experiments-restarting/imagenet-resnet50-sgd --model_name resnet50_sgd --optimizer "sgd" --gpu-id "0,1,2,3,4,5,6,7" --manualSeed 0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1 imagenet.py -j 16 -a "resnet101" --data /datasets/imagenet --epochs 90 --schedule 31 61 --gamma 0.1 --lr 0.1 --train-batch 32 --test-batch 25 -c ./experiments-restarting/imagenet-resnet101-sgd --model_name resnet101_sgd --optimizer "sgd" --gpu-id "0,1,2,3,4,5,6,7" --manualSeed 0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1 imagenet.py -j 16 -a "resnet152" --data /datasets/imagenet --epochs 90 --schedule 31 61 --gamma 0.1 --lr 0.1 --train-batch 32 --test-batch 25 -c ./experiments-restarting/imagenet-resnet152-sgd --model_name resnet152_sgd --optimizer "sgd" --gpu-id "0,1,2,3,4,5,6,7" --manualSeed 0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1 imagenet.py -j 16 -a "resnet200" --data /datasets/imagenet --epochs 90 --schedule 31 61 --gamma 0.1 --lr 0.1 --train-batch 32 --test-batch 25 -c ./experiments-restarting/imagenet-resnet200-sgd --model_name resnet200_sgd --optimizer "sgd" --gpu-id "0,1,2,3,4,5,6,7" --manualSeed 0
```

# SRSGD for ImageNet using short training

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1 imagenet.py -j 16 -a "resnet50" --data /datasets/imagenet --epochs 80 --schedule 31 51 --restart-schedule 40 80 80 --gamma 0.1 --lr 0.1 --train-batch 32 --test-batch 25 -c ./experiments-restarting/imagenet-resnet50-short-training-srsgd --model_name resnet50_srsgd --optimizer "srsgd" --gpu-id "0,1,2,3,4,5,6,7" --manualSeed 0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1 imagenet.py -j 16 -a "resnet101" --data /datasets/imagenet --epochs 80 --schedule 31 56 --restart-schedule 40 80 80 --gamma 0.1 --lr 0.1 --train-batch 32 --test-batch 25 -c ./experiments-restarting/imagenet-resnet101-short-training-srsgd --model_name resnet101_srsgd --optimizer "srsgd" --gpu-id "0,1,2,3,4,5,6,7" --manualSeed 0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1 imagenet.py -j 16 -a "resnet152" --data /datasets/imagenet --epochs 75 --schedule 31 51 --restart-schedule 40 80 80 --gamma 0.1 --lr 0.1 --train-batch 32 --test-batch 25 -c ./experiments-restarting/imagenet-resnet152-short-training-srsgd --model_name resnet152_srsgd --optimizer "srsgd" --gpu-id "0,1,2,3,4,5,6,7" --manualSeed 0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1 imagenet.py -j 16 -a "resnet200" --data /datasets/imagenet --epochs 60 --schedule 31 46 --restart-schedule 40 80 80 --gamma 0.1 --lr 0.1 --train-batch 32 --test-batch 25 -c ./experiments-restarting/imagenet-resnet200-short-training-srsgd --model_name resnet200_srsgd --optimizer "srsgd" --gpu-id "0,1,2,3,4,5,6,7" --manualSeed 0
```