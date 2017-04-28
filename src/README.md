# Source Code Repository
## Vanila GAN with fc layers
* Only 2 fully-connected layers for both discriminator and generator.
* Use [Xavier initializer](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) to accelerate training process.
* Gif of training result:

<img src="https://github.com/TengdaHan/GAN-TensorFlow/blob/master/figure/2fc-mnist.gif" width="256px">

* Figure of loss functions:

![](https://github.com/TengdaHan/GAN-TensorFlow/blob/master/figure/2fc_mnist_dloss.JPG) ![](https://github.com/TengdaHan/GAN-TensorFlow/blob/master/figure/2fc_mnist_gloss.JPG)
### Instruction
To train the model, run the command: 
```python train.py```

Output images will be saved in ```src/output/``` directory.
Tensorboard log files will be saved in ```src/tmp/``` directory.
### Lessons learned
* Understanding how to [share variables](https://www.tensorflow.org/programmers_guide/variable_scope) in TensorFlow is important when training GANs. In the training, we create two discriminators for both MNIST images and generated images like:
  ```
  D_real, D_real_logits = discriminator(X)
  D_fake, D_fake_logits = discriminator(G)
  ```
  We must ensure these two discriminators use same weights and biases, i.e. they should be the same discriminator.
  
* When training GAN, we should freeze discriminator when training generators, and vice versa. In TensorFlow, we can specify the variables to be trained for optimizers, like the ```var_list``` here:
  ```
  tvar = tf.trainable_variables()
  dvar = [var for var in tvar if 'discriminator' in var.name]
  gvar = [var for var in tvar if 'generator' in var.name]

  d_train_step = tf.train.AdamOptimizer().minimize(d_loss, var_list=dvar)
  g_train_step = tf.train.AdamOptimizer().minimize(g_loss, var_list=gvar)
  ```
  
* [Xavier initializer](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) helps accelerate the training process.

## Vanila GAN with ConvNets
* Use small scale convolution networks for both discriminator and generator.
* Use [Xavier initializer](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) to accelerate training process.
* Gif of training result:

<img src="https://github.com/TengdaHan/GAN-TensorFlow/blob/master/figure/conv-mnist.gif" width="256px">

### Instruction
Ensure to specify model name 'conv' in the last row of [train.py](https://github.com/TengdaHan/GAN-TensorFlow/blob/master/src/train.py), like: 
```train(logdir='1', model='conv', batch_size=64)```

Then run command:
```python train.py```
