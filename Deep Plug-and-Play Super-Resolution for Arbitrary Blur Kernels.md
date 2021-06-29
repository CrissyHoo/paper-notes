# Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels

plug-and-play是一个modelbase的方法，也就是说，它不用训练。

如题所示，他能够对任意的一个blur kernel进行操作。

dnn-based methods are designed for bicubic degradation.

plugandplay方法是modular structure，也就意味着它的结构是十分灵活的，可以很方便地插入denoiser prior。

本文提出的模型很好地利用了blind deblurring methods(暂时有疑问)

为了optimize the new degradation induced energy function，我们还是讲这个算法和variable splitting technique结合起来了。其实这个更像是我读的上一篇文章的延伸，这里我们可以plug in任意一种super-resolver prior，而不仅仅是denoiser prior。

#### Intro

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

#### related work

首先交代了srcnn那一系列的东西，然后在超越bicubic degradation之后的一些发展。

an **accurate estimate of the blur kernel** is more important than **sophisticated image prior**. 这两者有什么区别呢？

*什么是image prior？*

[自然图像先验与图像复原_zbwgycm的博客-CSDN博客_图像先验](https://blog.csdn.net/zbwgycm/article/details/81187774)

plug and play模型的思路

unroll the energy function by variable splitting method and replace the prior associated subproblem by any off-the-shelf gaussian denoiser.

