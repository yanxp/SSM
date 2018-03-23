# SSM
Towards Human-Machine Cooperation: Evolving Active Learning with
Self-supervised Process for Object Detection

### License

SSM is released under the MIT License (refer to the LICENSE file for details).

### Citing SSM

If you find SSM useful in your research, please consider citing:

    @article{wang18ssm,
        Author = {Keze Wang, Xiaopeng Yan, Dongyu Zhang, Lei Zhang, Liang Lin},
        Title = {{SSM}: Towards Human-Machine Cooperation: Evolving Active Learning with
Self-supervised Process for Object Detection},
        Journal = {Proc. of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        Year = {2018}
    }

### Dependencies 

The code is built on top of R-FCN. Please carefully read through py-R-FCN and make sure py-R-FCN can run within your enviornment.

### Datasets/Pre-trained model

1. In our paper, we used Pascal VOC2007/VOC2012 and COCO as our datasets, and ResNet-101 model as our pre-trained model.

2. Please download ImageNet-pre-trained ResNet-101 model manually, and put them into $SSM_ROOT/data/imagenet_models

### Usage

1. training
Before training, please prepare your dataset and pre-trained model and store them in the right path as R-FCN.
You can go to ./tools/ and modify train_net.py to reset some parameters.Then, simply run sh ./train.sh.

2. testing
Before testing, you can modify test.sh to choose the trained model path, then simply run sh ./test.sh to get the evaluation result.

### Misc

Tested on Ubuntu 14.04 with a Titan X GPU (12G) and Intel(R) Xeon(R) CPU E5-2623 v3 @ 3.00GHz.
