# LD-DETR：用於視頻時刻檢索和精彩片段檢測的循環解碼器檢測Transformer

### 語言

[English](../README.md) | [簡體中文](./README_simplified_chinese.md) | 繁體中文

## 摘要

> 視頻時刻檢索與精彩片段檢測旨在根據文本查詢找到視頻中的對應內容。現有模型通常首先使用對比學習方法來對齊視頻和文本特徵，然後融合並提取多模態信息，最後使用Transformer解碼器解碼多模態信息。然而，現有方法面臨幾個問題：（1）數據集中不同樣本之間重疊的語義信息阻礙了模型的多模態對齊性能；（2）現有模型無法有效提取視頻的局部特徵；（3）現有模型使用的Transformer解碼器無法充分解碼多模態特徵。針對上述問題，我們提出了用於視頻時刻檢索和精彩片段檢測任務的LD-DETR模型。具體而言，我們首先將相似度矩陣提取到單位矩陣以減輕重疊語義信息的影響。然後，我們設計了一種方法，使卷積層能夠更有效地提取多模態局部特徵。最後，我們將Transformer解碼器的輸出反饋到其自身中，以充分解碼多模態信息。我們在四個公共數據集上對LD-DETR進行了評估，並進行了廣泛的實驗，以證明我們的方法的優越性和有效性。我們的模型在QVHighlight、Charades-STA和TACoS數據集上的表現優於最先進的模型（State-Of-The-Art）。我們的論文可在[這裡](./paper_traditional_chinese.pdf)獲取。

## 數據集設置

下載數據集並將其解壓到數據目錄，如下所示：

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

## 環境設置

創建 conda 環境並安裝所有依賴項，如下所示：

```
# 創建 conda 環境
conda create --name ld_detr python=3.8
# 激活環境
conda actiavte ld_detr
# 安裝 pytorch
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 pytorch-cuda=12.4 -c pytorch -c nvidia
# 安裝其他包
pip install -r requirements.txt
```

## 訓練

### QVHighlights

在 QVHighlights 上進行訓練如下：

```
bash ld_detr/scripts/train_qvhighlight.sh
bash ld_detr/scripts/train_qvhighlight_with_audio.sh
```

### Charades-STA

在 Charades-STA 上進行訓練如下：

```
bash ld_detr/scripts/train_charade.sh
```

### TACoS

在 TACoS 上進行訓練如下：

```
bash ld_detr/scripts/train_tacos.sh
```

## 許可

我們的代碼遵循 [MIT](../LICENSE.md) 許可。
