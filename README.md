# LD-DETR: Loop Decoder DEtection TRansformer for Video Moment Retrieval and Highlight Detection

### Languages

English | [简体中文](./useless_files/README_simplified_chinese.md) | [繁體中文](./useless_files/README_traditional_chinese.md)

## Abstract

> Video Moment Retrieval and Highlight Detection aim to find corresponding content in the video based on a text query. Existing models usually first use contrastive learning methods to align video and text features, then fuse and extract multimodal information, and finally use a Transformer Decoder to decode multimodal information. However, existing methods face several issues: (1) Overlapping semantic information between different samples in the dataset hinders the model's multimodal aligning performance; (2) Existing models are not able to efficiently extract local features of the video; (3) The Transformer Decoder used by the existing model cannot adequately decode multimodal features. To address the above issues, we proposed the LD-DETR model for Video Moment Retrieval and Highlight Detection tasks. Specifically, We first distilled the similarity matrix into the identity matrix to mitigate the impact of overlapping semantic information. Then, we designed a method that enables convolutional layers to extract multimodal local features more efficiently. Finally, we fed the output of the Transformer Decoder back into itself to adequately decode multimodal information. We evaluated LD-DETR on four public benchmarks and conducted extensive experiments to demonstrate the superiority and effectiveness of our approach. Our model outperforms the State-Of-The-Art models on QVHighlight, Charades-STA and TACoS datasets. Our paper is available at [here](./useless_files/paper_english.pdf). 

## Datasets Setup

Download datasets and extract them to the data directory as below:

```
ld-detr
├── data
├── features
│   ├── qvhighlight
│   ├── charades
│   ├── tacos
├── ld_detr
├── papers
├── READMES
├── standalone_eval
└── utils
```

## Environment Setup

Creating conda environment and installing all the dependencies as follows:

```
# create conda env
conda create --name ld_detr python=3.8
# activate env
conda actiavte ld_detr
# install pytorch
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 pytorch-cuda=12.4 -c pytorch -c nvidia
# install other packages
pip install -r requirements.txt
```

## Training

### QVHighlights

Training on QVHighlights as follows:

```
bash ld_detr/scripts/train_qvhighlight.sh
bash ld_detr/scripts/train_qvhighlight_with_audio.sh
```

### Charades-STA

Training on Charades-STA as follows:

```
bash ld_detr/scripts/train_charade.sh
```

### TACoS

Training on TACoS as follows:

```
bash ld_detr/scripts/train_tacos.sh
```

## LICENSE

Our codes are under [MIT](./LICENSE.md) license.
