
# Guided-Attention-Inference-Network (GAIN)#

![image](https://user-images.githubusercontent.com/50303550/136835409-81275524-73ea-4e05-bb0c-06611b617eac.png)

## Introduction ##

This is a Pytorch implementation of the GAIN algorithm presented in the following paper ([paper CVPR2018](https://arxiv.org/abs/1802.10171)):

## Requirement ##

* Python >= 3.7
* [Pytorch](http://pytorch.org/) >= v1.8.1
* [Tensorboard](https://www.tensorflow.org/tensorboard) >= v2.5.0

Ensure on PyTorch & NVidia websites you have the appropriate CUDA drivers and Cudnn version.

## Paper implementation, run instructions, key issues and results ##

![image](https://user-images.githubusercontent.com/50303550/136822833-933c235c-a6e4-44a5-87ac-12263fb0c218.png)

We implement 5th formula in two ways as the paper doesn't present concrete methodology & implementations details:
![image](https://user-images.githubusercontent.com/50303550/136823043-52df3d1c-e602-4db6-80f1-e6e760bcde87.png)

we also use [AutoAugment](https://arxiv.org/abs/1805.09501) with CIFAR10 policy implemented by [torchvision](https://pytorch.org/vision/stable/transforms.html?highlight=autoaugment#torchvision.transforms.AutoAugment)

The first way can be seen in a more authentic and corresponding approach to the paper (in our opinion):
For each of the labels we compute a separate masked image, thus for example if an image has 3 labels, we compute 3 masked images,
(3 heatmaps, 3 masks and e.t.c) and the loss for that image will be the average of their scores after forwarding them.
This implementation appear and used in the following scripts:
1. [the model](https://github.com/ilyak93/GAIN-pytorch/blob/main/models/batch_GAIN_VOC_mutilabel_singlebatch.py) batch_GAIN_VOC_mutilabel_singlebatch
2. [main run script](https://github.com/ilyak93/GAIN-pytorch/blob/main/main_VOC_multilabel_heatmaps_singlebatch.py) main_VOC_multilabel_heatmaps_singlebatch

It works only with batch size of 1 as in the paper.

The second way which is easier to implement especially for batch size greater then 1 is:
The masked image can be compuited w.r.t all of the labels at once, as much as its score,
then it is devided by the amount of labels as in the first approach. See the technical implementation details
for more understanding how it is done and how both approaches differ.
This approach is implemented in the corresponding scripts:
1. [the model](https://github.com/ilyak93/GAIN-pytorch/blob/main/models/batch_GAIN_VOC.py) batch_GAIN_VOC
2. [main run script](https://github.com/ilyak93/GAIN-pytorch/blob/main/main_GAIN_VOC.py) main_GAIN_VOC

In this approach you can run with batch size even greater then 1.

You can experiment with each one of them.
