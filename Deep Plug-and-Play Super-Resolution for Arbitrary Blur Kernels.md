# Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels 2020

plug-and-play是一个modelbase的方法，也就是说，它不用训练。

如题所示，他能够对任意的一个blur kernel进行操作。

dnn-based methods are designed for bicubic degradation.

plugandplay方法是modular structure，也就意味着它的结构是十分灵活的，可以很方便地插入denoiser prior。

本文提出的模型很好地利用了blind deblurring methods(暂时有疑问)

为了optimize the new degradation induced energy function，我们还是讲这个算法和variable splitting technique结合起来了。其实这个更像是我读的上一篇文章的延伸，这里我们可以plug in任意一种super-resolver prior，而不仅仅是denoiser prior。

### Intro

选一个恰当的degradation model是十分重要的。

目前被广泛使用的degradation model有两种

1. 先blur kernel然后downsample然后加noise，这个在前几篇论文中都提到了

这种方法的缺点就在于他模拟的是一种预先就知道的blur kernel，在现实生活中这几乎是不可能的（就是说kernel是已知的

还有一些工作是去以estimating blur kernel为目的的，但是他们代码不开源。

比如2013年的blindsr

2. 直接进行downsample，一般来说都是bicubic downsampler,这种比较简单，但是对于dnn解决sr问题有很大的帮助，同时，由于这个逻辑过于简单，在practical的过程中表现得并不好。

所以后来这个作者就专门写了一篇文章关于一个更加适应于practical use的degradation model。

综上，有两个issue就显得很重要了，①设计出不同的degradation model ②将dnn的作用发挥到极致，使dnn能够适应于不同的degradation model

这篇文章呢，就使用了第一种degradation方法，使用blind deblurring method来估计lr图像的blur kernel。deal with arbitrary blur kernels

所以就提出了DPSR

在Fourier域中，我们可以更好地应对blur distortion。

综上，本文提出的模型，不仅在degradation上做出了更practical的处理，并且在sr的阶段，提出了一个加强版的plugandplay方法。

### related work

首先交代了srcnn那一系列的东西，然后在超越bicubic degradation之后的一些发展。

an **accurate estimate of the blur kernel** is more important than **sophisticated image prior**. 这两者有什么区别呢？

*什么是image prior？*

[自然图像先验与图像复原_zbwgycm的博客-CSDN博客_图像先验](https://blog.csdn.net/zbwgycm/article/details/81187774)

prior被理解为自然图像本身的性质，就如文中所说的，可以从损失函数的角度得到感性理解，从MAP角度得到严谨的推导。

而再具体一点：image prior包含了图像的局部平滑性，非局部相似性，稀疏性。（但其实我觉得这些性质都非常地不好量化

**plug and play模型的思路**

unroll the energy function by variable splitting method and replace the prior associated subproblem by any off-the-shelf gaussian denoiser.

plug-and-play的思想在前几年中被很广泛的应用过（我的理解就是往网络中plug一些模块的思想）这些研究主要从这几个方面进行：

①不同的variable splitting algorithm。比如HQS, ADMM, FISTA，primal-dual

②不同的应用，Poisson denoising, demosaicking, deblurring, sr, inpainting

③不同类的denoiser prior。BM3D, DNN-based denoisers, combinations.

④theoretical analysis on the convergence from the aspect of fixed point [13, 37, 38] and Nash equilibrium [10, 16, 45]（没懂

目前的这些方法只是将高斯denoiser看成prior，然而prior不仅仅是这些。

### Method

首先对degradation model进行了一个新的定义。

我的理解：因为大家在downsampling的过程中用到的都是bicubic downsampling，因为这种方法是已知的，所以我们其实也可以将bicubicly downsampled image看成是clean image（没有noise的image）。

那我们的degradation model就对应了一个deblurring+denoiser，这些都有现存的一些方法可以被很好地利用，作者觉得这是一种优势。

*new degradation model新在哪里呢？将downsampling简化成了bicubic？*

然后就去定义了energy function。

对于discriminative learning method来说，他们参照的模型事实上就对应了一个这样的energy function，其中degradation model（也就是保真项）被implicitly定义在训练lr和hr之间的对应关系中。这也解释了为什么现存的在bicubic degradation上train的dnn based methods在real image上表现得并不好。

#### deep plug and play SISR

然后又是HQS+iterative solution（虽然不知道是怎么解的

迭代求解得到两个解，x的那个主要负责deblurring，z的那个主要负责denoise，在不停迭代中，使得图像更加完美

#### deep superresolver prior

他用了一个很简单的srresnet为基础，然后进行改进，主要改进的地方在于：

1. 新加入了noise level map作为input

2. 增加了feature map的数量，从64到96

3. remove BN(在前面有一个研究中就有证明去掉bn效果更好)

然后我们就可以开始训练这个模型，在训练之前需要从hr中得到lr，这个过程就体现了上面的第一点。

然后给出了一些训练srresnet+的细节。

#### Comparison

建议阅读原文

疑问：

iterative solution 在实际情况中是如何实现的？

为什么用FFT就可以减少blur distortion？

#### Experiment

##### 合成LR

blur kernel：gaussian blur kernel, motion blur kernel(在基础知识里有科普)，disk(out of focus) blur kernel. 文中给出了kernel的图示以及表示。

parameter setting：在iterative solution中，我们需要设置lambda和μ的值，

在这方面遵循这两个principle：

①lambda是固定的而且可以用sigma表示，我们可以把sigma乘以根号lambda，所以就可以忽略8式中的lambda，应该就是把lambda拿到式子外面去的意思？

![8式](https://i.loli.net/2021/06/29/GPM2C43WiBZKcjw.png)

②μ是根据这个循环非单减的，我们写他的倒数，这样就非单增了，阿巴阿巴阿巴，我们看下12式

![12式](https://i.loli.net/2021/06/29/RlgztEP1wJFm5MV.png)

不懂，怎么就间接地决定了每个循环中的μ的值了。。。

结论，lambda=1/3, 指数递减根号下1/μ

compared method：VDSR, RCAN（bicubic degradation），IRCNN+RCAN, DeblurGAN+RCAN(cascaded deblurring+SISR)，GFN, ZSSR(for blurry lr images)

什么叫take the blur kernel and noise level as input？

因为作者之前提到，需要先对hr图像做处理（上述处理），就这种方式来进行input嘛，那过去的bicubic degradation的方式不是也可以这样处理嘛，那也可以把这两个作为input啊。

看代码完了再说吧

quantitative results: 从实验结果中可以得出一些结论。其实觉得这是个病句。RCAN和VDSR在bicubic degradation的表现上有很大的差别，但是在complex degradation setting上二者的表现差不多。

先进行deblurring可以有效地提高模型性能

最后夸了一波DPSR, 因为他直接对energy function做处理

[cszn/DPSR: Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels (CVPR, 2019) (PyTorch) (github.com)](https://github.com/cszn/DPSR)

这个工作还挺有意思的，虽然还有挺多疑惑的地方。准备复现一下