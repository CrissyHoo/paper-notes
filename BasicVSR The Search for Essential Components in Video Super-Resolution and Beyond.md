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

对**information-refill mechanism**做一个说明：有一类方法，它将video frames分成independent intervals，分类的依据就是，觉得有些是keyframes，另一些不是。对不同的这些frames处理方式也不同。IconVSR的一些方法是以上概念的延续，但是过往的方法对不同的interval是分开处理的，这里不是。

connecting the intervals through the **propagation branches**（所以是咋实现的呢

#### Methods









