# Deep Unfolding Network for Image Super-Resolution

准备读这篇是因为读的上一篇里面有一个疑问，关于为什么nearest neighbour interpolation会造成shift

上一篇文章提出了一个degradation model，各种模型在经过这样的model处理之后在实际应用中会得到更好的效果。并且对未来做出了展望：

We further argue that the IQA metric for SISR should also be updated with new image degradation types, which we leave for future work.

IQA metric并不能很好地衡量一个图片是否restore得“好”，可以提出一些新的衡量指标。但是我觉得这好像并不能单独作为一个研究，对于训练一个更“好”的模型并没有帮助，只是能够更加直观准确的用一个数据去衡量结果。（人眼可以直接做到

------

Deep unfolding network

*啥叫unified MAP(maximum a posteriori)?*

[Prior 、Posterior 和 Likelihood 的理解与几种表达方式_kww_kww的博客-CSDN博客](https://blog.csdn.net/kww_kww/article/details/52527888)

这个还得看其他的论文。

复习一下先验和后验：它的基础是贝叶斯公式。

给一个场景，一个袋子里有3只红球，2只白球，不放回摸取。

先摸一个红球，再摸一个红球，已知第二次摸到了红球，求第一次是红球的概率。

先验概率（prior）指的是根据**以往的经验和分析**，在还没有做实验，还没有采样的时候就可以得到的概率。就像我们通常所说的概率。比如，上面第一次摸到红球的概率。

后验概率（posterior）指的是某件事情已经发生，想计算这个时间的发生是由于某个**条件**引起的概率，比如，已知第二次摸到了红球，求第一次是红球的概率。

那结合到本文中的问题上，先验就是我已经知道了你是根据什么方法来降质的（bicubic等）。model-based的方法会比learning-based更灵活（使用map），为了理解图像处理中的后验我们还是需要去看map framework是什么样的。这个问题暂时先放着。

*关于maximum a posterior(最大后验概率模型)*

[小样本学习（Few-shot Learning）之特征提取器-最大后验概率估计（MAP）、Wasserstein距离、最优传输-Sinkhorn算法_gcheney的博客-CSDN博客](https://blog.csdn.net/gcheney/article/details/108442861)

map和最大似然估计是相对的，map是根据经验数据获得对难以观察的量的点估计，这一点和最大似然估计类似。但是**不同的是**，最大后验估计想要最大的量是不同的，融入了先验分布，看下面的例子：

首先估计指的是什么？首先有模型，数据的概念。我们用模型来拟合实际情况中研究对象表达出的一些数据。但是一个模型的确定是需要很多参数的，最大似然估计做的事情就是我得到了很多的数据，用这个数据去估计这个参数，**最大似然**就是针对给出的数据得到“最有可能”拟合的这个数据的模型（参数）。最大似然最大化的是：P（X|theta），最大后验想要求的是使得P(X|theta)P(theta)最大的theta，也就是还需要P(theta)的值也比较大。但是这个逻辑我觉得很奇怪，不知道怎么感性解释。

我们把这个式子稍微变形一下，记待求式为L，就可以得到，P(theta|X)=L/P(X)，因为X是确定的，所以其实就是在最大化P(theta|X)。因为X是结果，是已知的，所以我们说这个是后验概率。

------

本文提出的unfolding network利用了learning-based method同时还有model-based method，model-based method 描述的应该就是一些传统的图像超分的方法。他通过**half-quadratic splitting algorithm**（关于这个算法，看另一篇10年论文，我们在下一段进行简要解释）展开MAP inference

[HQS——Half Quadratic Splitting半二次方分裂 - ostartech - 博客园 (cnblogs.com)](https://www.cnblogs.com/wxl845235800/p/10734866.html)

（没看懂，，迭代求解那块）

论文把我们待解决的问题分成了两个子问题：data subproblem & prior subproblem   它兼有了两者的优点。

仍然提出sr问题的痛点是在研究中我们使用的degradation方法可能并不能很好地应用在实际情况中

### Intro

downsampler: 一般来说就是对于每个s*s的patch来说，取最upper-left的一个像素，丢弃其他的像素。

model-based方法受阻（bicubic degradation is mathematically complicated.

CNN在处理高斯噪声，模糊上的功能不够flexible

data term：enforce degradation consistency knowledge

prior term：guarantee denoiser prior knowledge(记住降噪器的先验知识？)

一些背景：

#### degradation models

classical degradation； bicubic degradation； 

直接下采样，不进行blur

bicubic downsample

说明了degradation model的发展，他们的发展让我们对于CNN based method的进一步研究是有意义的

#### Flexible SISR method

需要将scale factor， blur kernel and noise level都考虑在内

plug-play methods，plug the learned CNN prior into the iterative solution under the MAP framework 能够非常地灵活， handle various blur kernels, scale factors and noise levels.(在下一篇文章中会提到这个方法)

如何理解blind？

（还有疑问）blind表示，blur kernel等等一些参数在网络训练之前都是未知的。但是很奇怪的是，downsample的时候是你自己进行downsample的，你肯定会知道参数，只有一种情况，也就是你用的real image，这也就是说，blind=使用real image，好像有点问题

后文提到，我们可以从一些其他的地方，例如相机设置中得到blur kernel等等一些参数，从这个角度看，我们就在解决non-blind问题。

#### deep unfolding image restoration

可是什么叫unfold method，

deep unfold method的缺点：

prior subproblem解决的没有用deep CNN解决得好。

data subproblem没有用closed-form solution。

讲了一段fft(fast fourier tranformer)快速傅里叶变换



#### deep unfolding network

使用了那个half算法，我们得到了两个子问题，

网络的结构就是data module和prior module在不断堆叠，下面对这两个module进行更加详细的描述。

我觉得为了懂得所谓的data term和prior term，还是需要去看那个什么split算法的意思。或许可以具体看看那篇论文。

没太看懂data module。。data module中没有需要训练的参数（为什么）模型由于将data module和prior module分离得很好，所以模型泛化性能会更好。

prior module的目的在于，获得一张更加干净的图像（指没有噪音），他将datamodule的输出作为本模块的输入，然后进行去噪处理。

这个denoiser其实就是一个网络，ResUNet，是U-net和resnet的结合，然后讲了一下这个网络的结构。

clearer应该指的是图像分辨率的提高，但是图像分辨率提高了不代表没有噪声，cleaner指的是图像的噪声更小了。应该注意噪声和分辨率没有关系。

另一个部分是hyper-parameter module。相当于是一个控制台，它控制data module和prior module的输出。

##### End to end training

提到了loss function是如何设置的，还有一些其他的参数的设置，比如noise level，学习率是如何衰减的，训练的epoch设置等等。

#### Experiments

数据集：BSD68 

为了合成不同的lr图像，我们需要知道blur kernel和noise level这两个参数。 

实验过程中选择了12个有代表性的不同的blur kernel，有4个各向同性的不同宽度的高斯核，四个其他论文中出现的高斯核，4个motion blur高斯核。

后面继续分析了kernel robustness（不知道这是啥）然后就分别去研究对于每个blur kernel来说psnr的结果，

然后给出了psnr的结果，输入的图像是经过了不同的scale factor，noise level，blur kernel的组合的处理后的结果，从table中我们可以看出，usrnet表现得最好，那其实这些复杂的处理只是为了说明这个网络的泛化能力强，能够在不管什么样的情况下都获得好的结果。

因为现在的研究已经不仅仅局限于bicubic degradation。需要在更复杂的degradation中也获得好的效果。

#### 对于data module和prior module做进一步分析

usrnet是一个iterative method，所以我们来看一下在不同的iteration中，data和prior部分分别对hr的估计起到了什么样的作用。

通过实验：五个循环 得到的psnr结果会减少一些。data module的作用是减少blur kernel导致的degradation。p可以对高频信息进行挖掘。

这个usrgan是在scalefactor为4的位置上训练的，但是结果在scale factor为3的地方应用，仍能够应用得很好，说明这个模型的泛化性能很好。

首先要理解这个，他使用了快速傅里叶变换，前提条件是卷积is carried out with circular boundary condition.

[Circular/Linear Convolution 与 DFT_毛财胜的专栏-CSDN博客](https://blog.csdn.net/u012938704/article/details/79025175)

需要对边缘（boundary）做一个预处理。

#### Conclusion

这篇论文提出了一个usrnet，能够针对不同的degradation均恢复出比较好的效果。像前文提到过的，他有需要训练的部分，也有不需要训练的部分，这两个部分的结合使得它囊括了modelbased和learningbased的优点。

再次总结了data module的作用是clearer.prior module的作用是cleaner。

超参数模块则是对这两个模块的输出得到控制。

然后我们再来解决为什么读这篇论文的问题：

其实这篇文章没有刻意解决为什么nearest neighbour interpolation会造成shift？所以为什么这样的插值会导致我们需要进行一个shift呢？

原文：

> In order to downsample the HR image, perhaps the most direct way is nearest neighbor interpolation. Yet, the resulting LR image will have a misalignment of **0.5×(s − 1)** pixels towards the upper-left corner [49]. As remedy, we shift a centered 21 × 21 isotropic Gaussian kernel by 0.5×(s − 1) pixels via a 2D linear grid interpolation method, and apply it for convolution before the nearest neighbour downsampling.

 根据这篇论文的描述，其实就是取一个s*s的像素块，只保留它右上角的像素，

还有一点很奇怪的是，为什么他说要downsample，但是最直接的方法是最近邻插值，，，这个不是用来上采样的吗。。。