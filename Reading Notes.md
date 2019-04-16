Reading Notes

现在真的是心急如焚，

主要是找到了两篇github，我都star了一下，第一篇是SCNN，第二篇是lanenet

两个都有基于tensorflow的教程，然而他们需要安装很多python的包？

我用docker来训练，怎么才能安装上？？？？？？？？？？？？？？？？？？？？？

找了一些有关于import的解释：

import xx导入模块对于模块中的函数，每次调用需要“模块.函数”来用。
from xx import fun 直接导入模块中某函数，直接fun()就可用。
告诉你大法：from xx import * 该模块中所有函数可以直接使用。





## 4.9假期后开始工作

### 现在我要在自己的电脑上实现一下lanenet

关于论文的事情，我想还是等实现出来再来看一下吧，论文我已经下载下来了，<https://arxiv.org/abs/1802.05591>

首先重新搞一个虚拟环境，作为这次配置的基础环境，对了，我是对照着<https://github.com/MaybeShewill-CV/lanenet-lane-detection>来实现的

```
cd ~/lanenet1
virtualenv --system-site-packages -p python3 venv
source ~/lanenet1/venv/bin/activate
deactivate
```

现在文件夹lanenet1就是我这次设置的虚拟环境了，lanenet代表要进行的是lanenet的调试，1代表第一次调试

第二步就是安装必要的系统：可以用到教程中给出来的命令

```
pip3 install -r requirements.txt
```

不过我性格谨慎，我自己一个个安装的

```
pip3 install easydict==1.6
pip3 install opencv_python==3.4.1.15
pip3 install glog==0.3.1
pip3 install numpy==1.13.1
pip3 install matplotlib==2.0.2
pip3 install scikit_learn==0.19.1
pip3 install tensorflow==1.10.0
pip3 install tensorflow_gpu==1.10.0
```

其实安装大部分的包的时候都没有问题，但是安装numpy==1.13.1出现了一个版本不匹配的问题，我先把截图放在这里，等到以后遇到问题时候再来看一下

![](tyima/c1.png)

再把这个包下载下来

用到的指令是

```
git clone https://github.com/MaybeShewill-CV/lanenet-lane-detection
```

当然也可以直接打开网页下载，不过下载下来的是压缩包

克隆太慢，改成下载，下载也下载不下来，什么鬼哦

现在用另外一个电脑下载，再把压缩吧考到这个电脑上吧，服了

运行命令出错

```
python test_lanenet.py --is_batch False --batch_size 1 --weights_path path/checkpoint --image_path data/tusimple_test_image/0.jpg
```

啥情况？咋办？

## ，我要试试另外一个SCNN

首先先去<https://github.com/cardwing/Codes-for-Lane-Detection>上下载工程

然后启动虚拟环境

去global_config.py里把GPU数量改成0

然后下载好vgg16.npy和vgg19.npy，到底是选择哪个啊，一打开链接好多文件啊，

提前训练好的权重是个什么玩意？？？？？？

这些文件是哪个？ooo权重文件搞定了

图像文件是什么鬼

用绝对路径！！！！！！

我的电脑太菜了，凉了，必须要用集群了

测试命令

```
CUDA_VISIBLE_DEVICES="0" python test_lanenet.py --weights_path path/culane_lanenet_vgg_2018-12-01-14-38-37.ckpt-10000 --image_path demo_file/list.txt --save_dir save --use_gpu=0
```

这个命令成功啦

下面在进行train之前，我们来试试以前的那个lanenet吧

## lanenet重新测试

good good，原来是之前的权重文件不对

好了，这下成功了

运行成功的命令是

```
python test_lanenet.py --is_batch False --batch_size 1 --weights_path path/tusimple_lanenet_vgg_2018-10-19-13-33-56.ckpt-200000 --image_path data/tusimple_test_image/0.jpg
```

结果比上一个更好啊，更是我想要的那种

好的，接下来好好研究研究这个代码的，训练这个model吧

## 4.13好了从今天起应该学些自己的理论了

明天去valse，晚上聚餐，啥时候学这个理论啊

还要考科三，看妇联，没时间了啊

#### 语义分割

## 按分割目的划分

- ### 普通分割

  将不同分属不同物体的像素区域分开。 
  如前景与后景分割开，狗的区域与猫的区域与背景分割开。

- ### 语义分割

  在普通分割的基础上，分类出每一块区域的语义（即这块区域是什么物体）。 
  如把画面中的所有物体都指出它们各自的类别。

- ### 实例分割

  在语义分割的基础上，给每个物体编号。 
  如这个是该画面中的狗A，那个是画面中的狗B。

### 关于lanenet的论文

将车道检测问题改成一个实例分割问题

把整个网络的训练分成两个分支，它们分别是车道分割分支和车道嵌入分支

- 车道分割分支是属于语义分割的范畴，它的作用就是把整个图片分成两类，背景和车道
- 车道嵌入分支是在完成车道分割分支后，进一步分配给每个车道不同的ID，采用聚类损失函数来训练，
- 这样做的好处是可以处理任意数量的车道

做完这些之后，还要把聚类出来的每一条车道进行参数化表示，以前的人们都用曲线拟合，，，，等等还有其他的一些方法，但是都不好，

我们的解决方法，在对曲线进行拟合之前对图像应用透视变换，

- 以前透视变换的参数都是fixed固定的，我们的透视变换的参数是给了一个网络进行训练的
- 对于每一条车道都用三次多项式进行曲线拟合

##### 那么来看下实例分割的部分吧

实例分割=语义分割+聚类

看图2，就可以看出来，一共两个分支，嵌入分支进行像素嵌入，输出为嵌入向量，可以用于后面的聚类，语义分割分支进行二值化表示，最后两个综合起来就是实例分割，可以标记出每条车道的ID

语义分割：做语义分割的时候注意到的一些问题应该是这样的，

- 做标签的时候是把每个车道线的像素相连得到标签，这样做的好处是可以清楚一些断断续续的干扰
- 第二个是对loss function作出的一些改变，使用了boundedinverse class weight对loss进行加权，这样做的好处是为了解决样本分布不平衡的问题，也就是背景的像素点远多于车道线的像素点https://www.jianshu.com/p/c6d38d648509

实例分割：首先那种detect-and-segment的方法只适用于那种紧凑的图像compact objects，因此我们用one-shot的方法

另外一个分支：嵌入分支，是为了输出每个像素的嵌入值，是使属于一条车道的像素点之间的距离变得很小，而属于不同车道的像素点距离很大，这样做了之后，属于一个车道上的像素点会聚集在一起

- 实现的方法？用到两个公式，Lvar和Ldist，
- Lvar公式的作用是拉近属于同一车道线的像素点，就是如果像素点的向量与车道均值向量相差在某个threshold的话，就会更新，拉近两者之间的距离
- Ldist公式的作用是推远不同车道均值向量，就是如果两个车道线的均值向量小于某个threshold的话，公式更新，推远两个车道的均值向量
- 总的LOSS是，L=Lvar+Ldist
- 做完这个过程之后就会得到一个图像，这个图像的特点就是不同车道间的距离是大于d的而一个车道的距离是小于v的

聚类，接下来的步骤就是聚类了

- 默认上面的loss方程中，d>6v

- 首先使mean shift算法来避免将一些分离点分到其他的簇群中，选中一个离群点，这个是让分离点离簇群中心近一些<https://www.cnblogs.com/xfzhang/p/7261172.html>
- 然后在做一个thresholding算法，随机选一个嵌入点，然后以2v为半径，选中所有属于这个半径内的像素点，重复这个过程直至所有的嵌入点都分配在这个车道上面

网络模型结构，lanenet网络结构模型基于ENet

- ENet网络有两个分支，加密encoder分支和解码decoder分支，五个步骤，步骤123为encoder分支，45为decoder分支
- lanenet也有两个分支，segmentation branch和embedding branch，两个分支共享12步骤，345步骤各自训练
- s branch输出一个单通道二值图像，e branch输出多通道图像

## 接下来干嘛，还有一段时间才能吃饭，搞一下那晚的github吧

首先见个文件夹，专门用于放这些东西github1

```
cd github1/
git clone https://github.com/Phosphenesvision/Reading_Notes_about_DL.git
```

好像还有另外一种ssh的传输协议，等我有空了搞搞这个东西，下面介绍一些命令的作用

```
cd Reading_Notes_about_DL/    ###进入某个project的文件
git pull   ###拉下来，保持与云端同步
git branch   ###查看在哪个分支下面
git checkout -b zzp origin/zzp	###第一次去其他分支
git checkout zzp	###去其他分支
然后在相应的分支下做修改
git add Spatial\ As\ Deep\:\ Spatial\ CNN\ for\ Traffic\ Scene\ Understanding.pdf	###添加文件，如果是删除文件后面也要加名字
git add .	###只修改不上传新的东西
git commit -m "上传一篇有关于SCNN的论文"	###对这个操作作出一点解释
git status	###查看现在的状态
git push origin master	###把本地修改提交到master分支上，输入用户名密码
git push	###同上
```

本来想协议下ssh传输的教程，结果呵呵，发现官网上给的教程很全，行吧，去官网上查看就行了，在右上角头像那里，点进setting去就ok了