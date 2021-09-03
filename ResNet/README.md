
## 使用

下载数据集
[cifar](http://www.cs.toronto.edu/~kriz/cifar.html)

## 思路

将残差模块独立，问题在于另一条支流的x
这个x应该不是相同的x，而是预估的x，而预估的x在最后才越来越接近，因此应当是记录历史的x。
这个问题需要继续读论文发现

!! 有个关键的问题，之前没有注意，两条数据流的输入都是相同的，因此都是本次的输入

其他的单元都是之前熟悉的，因此不作讨论

首先的确是相同的维度，因为这样不涉及复杂的转换



## param 

记录参数的设置

SGD
batch size 128
weight decay of 0.0001  
momentum of 0.9

not use dropout 




 Then we use a stack of 6n layers wi

这个n的含义

We use a weight decay of 0.0001 and momentum of 0.9, and adopt the weight initialization in [13] and BN [16] but with  no  dropout.   These  models  are  trained  with  a  mini- batch size of 128 on two GPUs.  We start with a learning rate of 0.1,  divide it by 10 at 32k and 48k iterations,  and terminate training at 64k iterations, which is determined on a 45k/5k train/val split. We follow the simple data augmen- tation in [24] for training: 4 pixels are padded on each side, and  a  32×32  crop  is  randomly  sampled  from  the  padded image or its horizontal flip.  For testing, we only evaluate the single view of the original 32×32 image.

## 记录

- 9.1 
  - 实现最少层，学习率衰减 acc：0.8042      
TODO: ~~多GPU训练~~   
在尝试多GPU训练时，有报错情况，暂时没有头绪
- 9.2 
    - 单元化
    - 增加BN：We adopt batch normalization  (BN)  [16]  right  after  each  convolution  and before activation
    - 数据增强：We follow the simple data augmen- tation in [24] for training: 4 pixels are padded on each side, and  a  32×32  crop  is  randomly  sampled  from  the  padded image or its horizontal flip. 
    - 完成多GPU训练，使用DataParallel
    - 将损失函数进行修改-虽然使用了上述trick，但准确率并没有明显提升
- 9.3
  - 发现重大问题：重复层的使用——虽然层的参数相同，**但在forward中使用重复层不会有效**
  - 完成重复层修改，通过`nn.moduleList`   
TODO: ~~参数初始化方式 ~~
    > 实际上，Kaiming初始化已经被Pytorch用作默认的参数初始化函数 acc: 0.8333