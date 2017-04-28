# Source Code Repository
## Introduction
* Only 2 fully-connected layers for both discriminator and generator
* Use [Xavier initializer](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) to accelerate training process.
* Gif of training result:

<img src="https://github.com/TengdaHan/GAN-TensorFlow/blob/master/figure/2fc-mnist.gif" width="256px">

* Figure of loss functions:

![](https://github.com/TengdaHan/GAN-TensorFlow/blob/master/figure/2fc_mnist_dloss.JPG) ![](https://github.com/TengdaHan/GAN-TensorFlow/blob/master/figure/2fc_mnist_gloss.JPG)
## Instruction
A vanila GAN for MNIST dataset. To train the net, simply run in command:

```python train.py```
 
