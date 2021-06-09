# Designing a Practical Degradation Model for Deep Blind Image Super-Resolution

degradation model的作用。这里指的就是通过这个模型来获得的sr模型的输入。

blur这样的一些非常常规的degradation工作固然能起到一定的效果，但这是远远不够的。

本文就设计了一种degradation模型，他结合了randomly shuffled blur, downsampling and noise degradations

*什么是各向同性和各向异性的高斯核？*

高斯模糊指的是一个图像与二维高斯分布的概率密度函数做卷积

[高斯模糊浅析_weixin_34187822的博客-CSDN博客](https://blog.csdn.net/weixin_34187822/article/details/93146301)

上面的博文中对各向同性的高斯模糊做出了解释。各向同性指的是高斯核往每个方向的变化都是一致的。

各向异性的高斯模糊对应指的就是每个方向的变化是不一致的。

[isotropy与anisotropy?各向同性与各向异性滤波？_thankyvision-CSDN博客](https://blog.csdn.net/seasermy/article/details/97887632)

他们两个模糊的效果有些许不同。

Blind ESRGAN 

这个问题提出了一个degradation model，使得经过这个模型处理过后的图像能够让一般的sr模型表现出更好的应用性。

*degradation prior是什么？*

这里的prior翻译成“先验”，指的是降质先验，包括降质的类型，方式，强度等，好比下采样，如果我知道是bicubic或者已知的采样核，这就是降质先验（其实还是不是特别懂，就是模型先去学习prior的信息吗？）

过往的一些研究中，对于blur和noise都是以一个unified的形式处理，而不是cascaded framework

这篇论文的目的在于设计一个在实际情况下能够得到真正应用模型

### **Methods**

首先介绍了现存的超分研究方法以及他们存在的问题，并对这些问题的改进提供了简要的思路。

然后说明本文的内容：提出一个degradation model，这个model着重从blur，downsampling, noise, random shuffle strategy这几个方面去设计。

#### Blur

作者做了两个高斯模糊的处理，各向异性的高斯模糊和各向同性的高斯模糊

具体的如何被两个blur operation处理在后面的部分有讲。

目的：degradation space of blur can be greatly expanded

#### Downsampling

最直接的方式：近邻插值，但是近邻插值会导致一些问题，比如misalignment.

shift a centered 21 × 21 isotropic Gaussian kernel by 0.5×(s − 1) pixels via a 2D linear grid interpolation method线性网格插值法，将这个kernel做了一个位移。

nearest neighbour downsampling: 最近邻下采样

论文中对于nearest，bicubic和bilinear的方法的下采样都尝试了

还有一个down-up的方法，我们用这四种方法对hr进行下采样

#### noise

广泛应用的：高斯噪声  本文中提到的模型除了高斯噪声外，还考虑了JPEG压缩噪声和camera sensor noise

高斯噪声  

给出了产生高斯噪声的步骤

JPEG压缩带来的噪声

quality factor的作用+在degradation model中对JPEG带来的degradation的处理：应用两次JPEG压缩，这两次执行的概率分别是0.75和1。

processed camera sensor noise

demosaicing; exposure compensation; white balance; camera to XYZ (D50) color space conversion; tone mapping; 

#### Random shuffle

shuffle一个image有很多种方式，比如blur，downsample，noise等等，他们可以以不同的顺序对image进行degradation处理，而本文中提到的degradation model就选择了一种以随机的顺序对图像进行处理的方式。

其中一种是下采样，在downsampling中我们提到downsampling有四种方式，在这里也是对这四种方式进行随机选择。

这样degradation space就会大很多。

然后解释了一下fig1,给出了一些本文的model中进行degradation的方法。

#### 一些附加说明

为了更好地理解新的degradation model，做出一些说明。

1. degradation model顾名思义，他最直接的作用是train a deep blind super-resolver with paired lr/hr images. 这样就打破了sr问题中关于数据集的壁垒。

2. 它不适合去模拟一个已经降质的image？

3. 这个model所创造出来的效果可能在实际情况下很少遇到，可能可以提高模型泛化能力

   4和5都在描述这个模型很厉害，然后，可改动性强，也很好改

#### 训练过程

这个模型能够生成**synthetic** image pairs，然后让过去提到的一些很经典的模型在这样的数据集上训练，就可以使那些模型获得更好的效果。 

然后列举了一些训练参数的设置

数据集：从100张DIV2K中生成了300张degradation后的图片，使用了三种不同的degradation 方法。和20张real world中拿到手就已经是降质了的图片。

然后是实验过程，实验在三种类型的degradation方法上进行。然后将其放在不同的模型上测试。

第一种：anisotropic gaussian blur with nearest downsampling by a scale factor of 4

第二种：anisotropic gaussian blur with nearest downsampling by a scale factor of 2 and subsequent bicubic downsampling by another scalue factor of 2 and final jpeg compression with quality factors uniformly sampled from [41,90]

第三种：本文提出的degradation model。

##### 跟其他的一些方法进行比较

对被比较的方法做出了一些介绍

LPIPS：learned perceptual image patch similarity另一个衡量超分效果的指标，越低越好

然后对实验结果进行了分析。表明了被提出方法的有效性。

对在真实情况数据集上实验结果的描述

因为这个时候，无法跟gt对比（没有gt），所以使用IQA（image quality accessment）作为衡量标尺。proposed method BSRGAN虽然没有在这个衡量中取得很好地成绩，但是从人眼来看它的效果是最好的。

future work：We further argue that the IQA metric for SISR should also be updated with new image degradation types, which we leave for future work.

#### 复现



