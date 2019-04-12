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

