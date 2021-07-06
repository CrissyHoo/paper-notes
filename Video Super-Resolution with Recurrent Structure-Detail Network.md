# Video Super-Resolution with Recurrent Structure-Detail Network

先来复习一下RNN：

一个序列当前的输出不仅与当下的输入有关，与前者的输出也有关。

![RNN_2](https://img-blog.csdn.net/20150921225357857)

RNN包含 input units, output units, hidden units

[循环神经网络(RNN, Recurrent Neural Networks)介绍_我和我追逐的梦~~~-CSDN博客_循环神经网络](https://blog.csdn.net/heyongluoyao8/article/details/48636251)

------

大部分的视频超分方法是在时间维度上使用neighbor frames对reference frame进行补充，相比较于recurrent-based method，前者的效率会更低，

本文就提出了一个recurrent vsr的方法，主要的思想是利用previous frame来还原当前帧。

将input拆解成structure and detail components，然后将这些喂入recurrent unit（很多个two-stream structure-detail blocks组成）

hidden state adaptation module：让当前的frame选择性地利用从**hidden state**中获取的信息。

什么叫two-stream block?

#### Intro

SISR：利用image prior来补偿图像丢失的细节。

MFSR: 怎么样从这么些additional temporal information中提取有价值的信息然后进行补偿

然后交代了一些背景。①用光流进行补偿②不显式地进行运动补偿，而是隐式地进行运动补偿，使用dynamic upsampling filter, the progressive fusion residual blocks 然后有一些文章是使用这种方式获得了很好地效果，但是redundant computation（因为需要对一个neighboring frame组进行缓存，这样计算量还是很大），所以为了减少计算量，有人就觉得可以applying recurrent connection to address the VSR task（就像nlp中的rnn，他处理的是一种sequence，其实视频帧序列也是一种sequence）

这些方法使用recurrent connection **in a streaming way**。序列前面的output/hidden state作为预测序列后面的frame的输入。

利用flexible supervision来恢复一些高频细节。（flexible supervision是什么）

我们发现hidden state每次捕捉到的画面的appearance都不一样，为了在hidden state的地方充分利用时间

对其中的hidden state进行了一些改进。因为发现它在capture scene‘s typical appearance的时候有一些问题，所以我们将hidden state当成一个historical dictionary 对待（这是个啥），并且计算reference frame和hidden state中每一个channel的相关性。

所以当前帧就可以更加关注有益的信息，也可以丢掉那些过时的信息。所以information fusion的部分就变得更加的robust了

#### Related work

本文工作将输入的帧拆分成structure和detail，然后将其输入到recurrent unit中（这个unit经过了精心设计），

structure and detail components not only suffer from different difficulties in high-resolution reconstruction but also take benefit from other frames in different ways（大概意思就是觉得这两个分量很有代表性吧

在videosr的问题上，跟SISR不同的是，sisr更多的是利用natural prior和self-similarity来对图像进行恢复，temporal information对于单张超分来说并不十分重要，但是对于videosr来说，就需要对帧间信息进行更多的处理，我们可以将处理vsr问题的思路分为两类，一类是显性地使用了motion compensation，另一类是隐形的使用了motion compensation。（因为肯定是需要进行motion compensation的）

##### Explicit motion compensation

显式的进行运动补偿的方法就遵循了一套pipeline：（虽然不是很懂具体是怎么实现的）motion estimation， motion compensation， information fusion， upsampling。

在这篇文章中：Frame-recurrent video super-resolution就有了使用recurrent的方法去恢复一个sequential frames。

“recurrent encoder-decoder module”?

就是专门对motion estimation和motion compensation设置对应的模块进行处理，这样就有因为这两个模块导致模型的计算量特别大的缺点

**Implicit motion compensation**

不直接地估计frames之间的motion，也不进行直接的对齐，而是直接跳到了fusion的步骤，希望设计出的fusion module可以充分利用frames之间的互补信息。然后列举了一些，其实不直观的进行这两个步骤的话，就需要用一些特殊设计的网络对这部分的信息进行采集。

本文也是用的隐式的去进行motion compensation的方法，不同的是我们将帧的信息拆成structure和detail两个部分，然后分别进行处理（在网络中进行传播）

感觉是照搬nlp里的内容，然后结合了一下图像的特性

**rnn在video-based task中的使用**

大概说RNN对于处理sequence类的信息很有优势。

#### Methods

上面也提到了，会把这个frame划分成两种信息，detail和structure，这两种信息会在SD-block中互相交互，这样就可以使structure更加鲜明，然后使在这个过程中能够恢复出更多的details。

除了上述信息，我们还可以从hidden state中获取上下文的信息，这样能对current frame进行更好的恢复。这个就能够让我们去很好地强调有用的信息，并且抛弃outdated信息。

然后我们分析一下那个示意图：在往前估计的过程中，t时刻模块接受上一个模块的输出，估计的structure和detail信息，t-1和t时刻的frame，然后经过一定的处理，估计出t时刻的frame，同时每个模块还会输出处理过后的structure和detail，可以给下一个模块使用。（里面向上的箭头 depth to space是嘛意思。。。就是上采样的意思

 **Recurrent Structure-Detail Network**

*recurrent unit*

首先我们来解释一下上面说的t时刻的模块，也就是recurrent unit，每个frame可以被分解出两方面的信息，一方面是structure，对应低频信息，也包含了一些motion between frames的信息；另一方面是detail，主要抓住的是“fine”高频信息and slight change in appearance。对这两种信息的处理方式不同，能从中获益的点也不一样，所以需要我们分开进行处理，

所以怎么提取structural information呢？我们对lr进行bicubic downsample和upsampling的处理，来提取，被记作S_t^{LR}（没办法敲行内公式orz）

怎么提取detail component呢？因为他两加起来等于原来的lr，所以现在用完整的lr减去上一步中的structure information就得到了detail information。

ps 也可以通过高通滤波器和低通滤波器分别获取不同频率的信息。

然后说在recurrent unit中使用了一个**symmetric architecture**，看原文和图，3.2节

![示意图](https://i.loli.net/2021/07/06/38bGY5IOs6Jg1mF.png)

这里的symmetric指得应该是对structure和detail的处理是对称的。

*Structure-Detail block*

那个蓝色的模块的内部结构。对sd模块的设置使用了很多方法来进行比较，

好像就是混乱连线哈哈哈哈哈，在resblock的基础上进行了一定的改进。

**Hidden State Adaptation**

对hidden state的改编利用

在rnn中，在时刻t的hidden state可以对t时刻之前的所有信息进行总结代表。

如果说将RNN和video结合，那我们期待的hidden state所能描述出的信息就是这个视频sequence所呈现出的**画面**随着时间的迭代有什么样的规律（在structure和detail这两个方面上）。

我们观察到一个现象，如果用之前那种方式（直接concat之前的hidden state和两个input frame，然后送进卷积层处理，这样有一些缺点，对每帧来说，他借鉴前几帧中信息的方式应该不一样（？？？））在hidden state中的不同通道对同一个scene的描述不一样，They should make different contribution to different positions of different frames（需要进行个性化定制）

hidden state的channel要怎么理解呢。。。

在上面所说的hidden state中不同channel给出的结果不一样的情况下，我们就不能对他们平等看待，而应该（类似attention机制）。对于跟current frame的相似度比较大的情况，应该更加重视。如果看起来差别很大，应该被丢弃。

以上是对基本情况的分析，然后说一说本文的实际操作。

本文提供的方法尽可能的从previous frame中只取那些**有用**的信息。

计算input frame和hidden state的相关性。

看原文，公式写的很明白3.3

element-wise multiplication就是逐个元素对应相乘

这样我们能获取更有效的hidden state

**loss function**

因为这个模型有两个stream，对两者的监督应该保持一致，所以在loss的设置上，也给这三者分别设置了对应的loss factor，一个是for structure component，一个是for detail component，还有一个for the whole frame

这三个factor分别也对应有权重，方便进行trade-off，因为一个视频有n frame，所以我们整体的loss是取这n个frame的平均。

[超分损失函数小结 - 江南烟雨尘 - 博客园 (cnblogs.com)](https://www.cnblogs.com/jiangnanyanyuchen/p/11884912.html)

每个factor具体使用的是charbonnier loss，它是l1 loss的变体，大概意思就是l2损失太过夸张，不适合超分，我们需要用更“敏感细腻”的l1 loss来处理像素损失。而charbonnier loss在根号下加了一个很小的常量，为了使数值稳定。

其中的frame loss就是对恢复后的frame进行总体的比较。

#### Experiment

ablation study研究的是SD block和hidden state adaptation的有效性

一些train的时候需要注意的地方，讲到了一些参数的设置，然后就是第一帧处理的时候没有previous frame，于是这些需要的参照都被设置为0.

通过对损失函数进行不同的tradeoff，也发现了一些规律

例行和sota进行一系列的比较，然后就是总结了一下这个模型的大体思路。



我的感悟就是，其实没有想到过原来RNN在视频这方面的应用很早就有了，我以为是近期transformer的大热导致nlp和cv领域的交互，他们两个的主要共同点就是sequence数据，需要从context中获取信息，nlp中需要处理语义，而视频中就需要对每个img进行信息提取，本文是从高频和低频的角度去提取的。接下来就加油复现吧！



