# BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond

读这篇是因为，引用了师兄推荐的那篇rsdn，采用了解决nlp问题的框架，内核还是对图像一些处理，使用implicit motion compensation，获得了很好的效果

呜呜呜希望训练顺利！！！再不要出什么岔子了，悲，不然信心真的要耗尽了...

看这个题目，作者对目前出现的一些超分中常用的components进行了细小的改进，然后分析他们对超分问题做出的贡献，试图找到什么样的模块能够更好地处理超分问题，达到更好的效果。

通过information refill mechanism的方式体现basicvsr的可拓展性。使用a coupled propagation scheme来利用集成的信息。BasicVSR和IconVSR能成为很nice的baseline。

#### Intro

SISR是利用图像本身的性质对图像进行upsampling，VSR就会更复杂一些，他需要集成很多联系得很紧密但是没有align好的frames的信息。

然后把现有的方法分了几类：

complex method：EDVR(显式地提取motion compensation，主要的特点是deformable convolution和multiple attention layer，虽然效果不错，但是太复杂了，参数量也很大)；RBPN（暂时没看过这个方法，使用了multiple projection modules去集成frames的feature，其实也不懂这里的projection指的是什么，这样就是运行时间太大了，其实效果也不算很好）；

VSR问题的主要困难的地方在于implementing and extending existing approaches, 阻碍了reproducibility and fair comparisons.（不能很好地根据现有的方法进行拓展，不同的方法之间不好进行比较（应该不是指单纯评价指标上的比较））

这个时候啊，我们就需要跳出来，不能陷在这个刷指标的怪圈里，重新考虑对VSR model的设计。从这个问题的本质出发，来进行研究。

一般来说解决这个问题分这么几个步骤：propagation， alignment， aggregate，upsampling

这里不太理解propagation指的是什么？OK后面解释了，感觉是在时间维度上，各个信息很冗余的帧的feature是怎么进行交互的。但是还是不是特别理解，因为没看过双向传播的文章。这里提到RSDN是单向传播的

一般来说aggregation方面大多数方法都是直接concat（感觉好随意啊，，，

upsampling部分清一色用的pixel-shuffle

但这样的分类让我们对VSR模型有了高一个维度的理解。

发现bidirectional propagation scheme可以最大程度地采集信息

使用optical flow-based method来进行相邻帧之间的feature alignment

在BasicVSR的基础上又新建出了IconVSR，用到了这样的module：information-refill和coupled propagation scheme（bidirection说的应该就是这个吧）

#### Related work

又从一个新的角度去看目前所有的VSR方法了，其实我觉得就是你是线性地处理数据，还是一大把抓的这种（整体的：**sliding window & recurrent**。

sliding window：就像有一个小窗口一样，在帧序列上进行滑动。看原文。原文没啥废话

recurrent：RSDN

bidirectional propagation coupled with a simple optical flow-based feature alignment suffice to outperform many state-of-the-art methods.

对**information-refill mechanism**做一个说明：有一类方法，它将video frames分成independent intervals，分类的依据就是，觉得有些是keyframes，另一些不是。对不同的这些frames处理方式也不同。IconVSR的一些方法是以上概念的延续，但是过往的方法对不同的interval是分开处理的，这里不是。（有点像注意力机制）

connecting the intervals through the **propagation branches**（所以是咋实现的呢

#### Methods

##### BasicVSR

在这个里面就对propagation里面提到的local啊，bidirection啊做了一些解释！（nice

在这个研究中我们把注意力放在光流和resblock上，

**propagation**

我们的研究对象是一个sequence，propagation模块的作用就是描述我们要怎么利用这个sequence的信息。有local， unidirectional, bidirectional propagation.

本文使用的是bidirectional propagation,所以我们首先来分析一下前两者的弊端

看原文

①local 就是用sliding window的方式，只是对序列的局部信息做了提取，对于distant information没有进行很好的提取。然后进行了一个实验来说明distant information的重要性

②unidirectional propagation

天啊这都是怎么想出来的，我能想到的最直接的方法就是提高算力，这样我们就可以对整个sequence一下子进行处理而不用考虑计算量爆炸的问题。

这种propagation的方法能够很好地解决①中提到的问题

这种方式就是能够让信息sequentially地从第一帧到最后一帧进行传播，也就是利用nlp的常用处理方法

因为是序列化的信息，这样第一帧没有preframe就没办法从序列中获取信息，同样的，这样看来最后一帧从序列中获得的信息就是最充分的。所以每一帧从序列中获得的信息都是imbalance的（这个很好解决啊哈哈哈，正向传播一次再反向传播一次

③bidirectional propagation

in which the features are propagated forward and backward in time independently.其实这个思路还蛮自然的。所以basicvsr就使用了这种方法，但论文中也没有说要怎么实现，光这样讲其实很抽象

##### Alignment

我理解的alignment就是这里有很多帧，这些帧中存在的画面是相同的，但是空间位置不同，对齐就是把这些画面对齐起来。

他的作用就是对齐内容相关性很强但是misaligned的image或者feature。为下一步的aggregate做准备，本文对比了主流的几种方法，最后选择了feature alignment。

①without alignment 

目前的一些recurrent方法通常就直接不做alignment的处理。（但是其实我不是很懂feature alignment需要怎么理解

这样的话其实弊端很大，因为会影响到后面aggregate的效果。所以这种方法肯定不行的

②Image Alignment

通过计算optical flow和warping的方式来进行alignment。大概结论就是feature alignment比image alignment更高效，所以我们应该进行feature alignment

③feature alignment

还是得去参考光流，然后对特征来进行对齐，然后在feature level上进行wrap。

##### Aggregation and upsampling

在basicVSR中用到的都是很基础的模块，（没有在这两个模块上做特殊处理

看原文，这部分没啥特殊的

##### summary

#### To IconVSR

在basicvsr的基础上，我们加上了两个新的模块，这样就变成了iconvsr，获得了更好的效果。

information-refill mechanism & coupled propagation

后者用来减小在传播过程中的误差。增强了aggregation的效果。

**information-refill**

这个部分是对特征有了更完美的提取

feature extractor和feature fusion都是只应用在某些选择出来的frame上，所以这个模块并没有给整个模型增加多少计算量

**Coupled propagation**

bidirectional setting，所以feature是向两个方向传播的，但是过往的一些方法里面，这两个方向的传播是相互独立的，这样不是很好，所以就把他们interconnected了，这样我们就可以得到质量更好的恢复图像

而且这种方法计算量也不是很大，因为只需要改变一下branch connection

#### Experiment

首先对dataset进行了说明，其实就是常用数据集，reds和vimeo。

然后也是几个常用的testset

（这个训练过程还蛮复杂的），spynet和edvr来做一个 预，用来做flow estimation和feature extractor，

实验其实还蛮简单的，因为用到的都是别人已有的模型，只是在方法上进行了创新，然后训练方法比较复杂，然后关于bidirection等等的具体操作就得自己看代码了

跟几个sota进行了比较

一点是对feature map中边缘像素的处理，因为我们有information-refill的机制，所以一些refill得到的额外的特征就可以对边缘进行补偿。

后面的就不说了，直接翻原论文，批注很详细。

就是这篇文章代码我有点搞不明白

不仅RSDN是个强大的baseline，这篇也是啊！！！冲冲冲！！！快想idea！！

真的可以好好借鉴一波nlp的典型方法。

然后要好好地利用光流。

再去研究一下transformer







