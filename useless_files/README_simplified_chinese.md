# LD-DETR：用于视频时刻检索和精彩片段检测的循环解码器检测Transformer

### 语言

[English](../README.md) | 简体中文 | [繁體中文](./README_traditional_chinese.md)

## 摘要

> 视频时刻检索与精彩片段检测旨在根据文本查询找到视频中的对应内容。现有模型通常首先使用对比学习方法来对齐视频和文本特征，然后融合并提取多模态信息，最后使用Transformer解码器解码多模态信息。然而，现有方法面临几个问题：（1）数据集中不同样本之间重叠的语义信息阻碍了模型的多模态对齐性能；（2）现有模型无法有效提取视频的局部特征；（3）现有模型使用的Transformer解码器无法充分解码多模态特征。针对上述问题，我们提出了用于视频时刻检索和精彩片段检测任务的LD-DETR模型。具体而言，我们首先将相似度矩阵提取到单位矩阵以减轻重叠语义信息的影响。然后，我们设计了一种方法，使卷积层能够更有效地提取多模态局部特征。最后，我们将Transformer解码器的输出反馈到其自身中，以充分解码多模态信息。我们在四个公共数据集上对LD-DETR进行了评估，并进行了广泛的实验，以证明我们的方法的优越性和有效性。我们的模型在QVHighlight、Charades-STA和TACoS数据集上的表现优于最先进的模型（State-Of-The-Art）。我们的[论文](./paper_simplified_chinese.pdf)可在这里获取。

## 数据集设置

下载数据集并将其解压到数据目录，如下所示：

```
ld-detr
├── data
├── features
│   ├── qvhighlight
│   ├── charades
│   ├── tacos
├── ld_detr
├── papers
├── READMES
├── standalone_eval
└── utils
```

## 环境设置

创建 conda 环境并安装所有依赖项，如下所示：

```
# 创建 conda 环境
conda create --name ld_detr python=3.8
# 激活环境
conda actiavte ld_detr
# 安装 pytorch
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 pytorch-cuda=12.4 -c pytorch -c nvidia
# 安装其他包
pip install -r requirements.txt
```

## 训练

### QVHighlights

在 QVHighlights 上进行训练如下：

```
bash ld_detr/scripts/train_qvhighlight.sh
bash ld_detr/scripts/train_qvhighlight_with_audio.sh
```

### Charades-STA

在 Charades-STA 上进行训练如下：

```
bash ld_detr/scripts/train_charade.sh
```

### TACoS

在 TACoS 上进行训练如下：

```
bash ld_detr/scripts/train_tacos.sh
```

## 许可

我们的代码遵循 [MIT](../LICENSE.md) 许可。
