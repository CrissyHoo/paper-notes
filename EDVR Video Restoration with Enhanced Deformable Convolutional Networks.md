# EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

感到羞愧，这么经典的文章我现在才读orz

解决了两个问题：

1. how to align multiple frames given large motions
2. how to effectively fuse different frames with diverse motion and blur

vsr中，因为画面是连续的，存在物体之间的移动，这里的align指的应该就是将连续帧上的物体对齐。

对齐了之后我们需要结合不同帧的信息，也就是我理解的fuse

handle large motion-> pyramid， cascading and deformable alignment module. 

在feature level就使用deformable convolutions进行帧对齐(coarse to fine manner)从一个粗糙到较完善的方式

我们提出了一个temporal and spatial attention fusion module，在时空域上都应用了attention mechanism

#### Intro

这个模型使用的数据集是REDS，更难训练出来。说明了一下这个数据集的官方性

要充分利用temporal redundancy。当时的最流行的方式是遵从这样一个很复杂的pipeline: **feature extraction, alignment, fusion, and reconstruction**.

alignment和fusion模块的设计是需要很多思考的。如果物体快速移动，fusion过大，这些都很难恢复，解决思路就是：align and establish accurate correspondences among multiple frames, and (2) effectively fuse the aligned features for reconstruction.对对齐之后的特征进行融合。

##### Alignment

像之前提到的那样，alignment的方法有：估计optical flow field（显式的alignment）；dynamic filtering or deformable convolution(隐式alignment)

如果motion很大的话，由只有一个scale的分辨率很难                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

##### Fusion

对齐帧后，需要对这对齐了的帧的特征进行融合。当前的一些方法基本上是使用卷积、recurrent network gradually fuse，rnn最开始是被用到nlp当中的，因为是对sequence进行处理，所以是不是可以考虑其实transformer也可以被用到里面。

然后指出当前的所有办法都没有consider the underlying visual informativeness.(没有考虑每帧的空间信息？)但是每帧的空间信息也不是那么好考虑的。。。



本文提供的方案：EDVR: PCD上文提到过，就是一个alignment module；fusion module TSA

PCD:为啥要叫金字塔呢，因为align的module的逻辑就是一个金字塔

①align features in lower scales with coarse estimation ②propagates the offsets and align features to higher scales(这里的scale指的是低的分辨率和高的分辨率吗)

在这个金字塔的后面还加了一层deformable convolution，来增强alignment的健壮性。

fusion module的作用就是将不同的aligned features的信息集成起来。

temporal attention：computing the element-wise correlation between the features of the reference frame and each neighboring frame.

这只是计算了相邻帧的相关性，然后我们会将根据这个相关性的协方差去对每一个相邻帧的像素赋予权重。（正好衡量了这个相邻帧的location对reference image有借鉴意义的程度）然后这些被赋予权重的frames就被convolve & fuse了，就处理完毕了。

在处理完这个之后，我们需要对spatial attention进行处理，spatial attention的作用就在于在本帧的每个通道的每个位置上赋予不同的权重，这样就利用到了cross-channel and spatial information。

#### Related work

讲了一下光流法，然后说EDVR没有用光流，使用的是implicit alignment。因为光流其实也handle不了很large的motion和很多blur的情况。

使用的是implicit alignment，用pcd的结构来处理large motion

在处理video blur的问题上，很多方法是直接将很多frame进行fuse，没有进行temporal alignment。因为blur会使temporal alignment非常的困难。

但是VDSR不这么做，他依旧尝试从将很多帧alignment后获取一些信息。但是先对所有的帧做deblurring的操作，减轻blur对alignment的影响。

**deformable convolution**（可变形的卷积，这个名字好奇怪）

能从neighbor中获取更多的信息。

TDAN这个模型也用到了。但是你看feature map就是一个挺抽象的东西，怎么align呢？感到迷惑。

这个卷积用在了PCD模块中，用来进行alignment

**Attention mechanism**

从不同的帧中获取信息来对feature map赋予不同的权重。

很好奇是怎么实现的，感觉每篇论文介绍注意力机制都是一个话术

#### Method

有一个中心的reference frame，从这帧开始，两边各n帧就是借鉴的帧，我们借鉴这些帧来恢复出中心帧。

①这些neighbor帧都与reference frame进行了alignment（through PCD）at feature level

②TSA module 负责对不同帧表达出的信息进行融合（fuse

③将这些fuse好的feature发送到reconstruction module里，这个模块的设计就会灵活一些。

④重建之后将原图进行上采样upsampling。sr=predicted image residual+upsampled image

##### alignment的具体过程，with pyramid， cascading and deformable convolution

使用deformable convolution去做alignment

PCD module：pyramidal processing and cascading refinement

为了生成固定level下的feature，看原文

##### 融合时空维的attention

不同的neighbor frame包含的信息量不同。

misalignment和unalignment会对reconstruction的效果有很大的影响。

所以要动态的做fusion（但是动态这个词就很灵性。。。

使用TSA module为每一帧都附上fusion所需要的的weight

相邻帧中，跟reference frame相似度越高的帧，在融合的时候被赋予的权重就应该更大。

我不太能理解这个从理论到实现的之间的逻辑的连续性。

##### 两个阶段的restoration

在输入图像有blur或distortion的时候，恢复效果不是那么好。

所以就用了2-stage去做恢复。

第一个stage就是用浅层的edvr网络做处理，这样对blur和distortion预先做了一个简单的处理，具体的过程我们来看实验

#### Experiment

数据集的使用的是REDS

PCD使用了5个residual block去提取特征。

5个连续帧作为输入，然后给了一些loss和optimizer的设计，就是套路啦

然后是跟sota的比较

分别从sr和deblurring两个角度找到了sota进行比较


what's flow magnitude?

that the smaller the motion is, the more informative the corresponding frames and regions are

还可以找作者要源码，真的很厉害诶