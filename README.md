# WPS-SAM: Towards Weakly-Supervised Part Segmentation with Foundation Models

Official PyTorch implementation of PPT from our paper: [WPS-SAM: Towards Weakly-Supervised Part Segmentation with Foundation Models](https://arxiv.org/abs/2407.10131). **ECCV 2024**.
Xinjian Wu, Ruisong Zhang, Jie Qin, Shijie Ma, Cheng-Lin Liu.

## What is WPS-SAM

![image](https://github.com/user-attachments/assets/f50cd1fe-2fd0-4102-8b1d-e29a983772fa)

Segmenting and recognizing diverse object parts is crucial in computer vision and robotics. Despite significant progress in object segmentation, part-level segmentation remains underexplored due to complex boundaries and scarce annotated data. To address this, we propose a novel
Weakly-supervised Part Segmentation (WPS) setting (as shown in the figure above) and an approach called WPS-SAM (as shown in the figure below), built on the large-scale pre-trained vision foundation model, Segment Anything Model (SAM). WPS-SAM is an end-to-end framework designed to extract prompt tokens directly from images and perform pixel-level segmentation of part regions. During its training phase, it only uses weakly supervised labels in the form of bounding boxes or points. Extensive experiments demonstrate that, through exploiting the rich knowledge embedded in pre-trained foundation models, WPS-SAM outperforms other segmentation models trained with pixellevel strong annotations. Specifically, WPS-SAM achieves 68.93% mIOU and 79.53% mACC on the PartImageNet dataset, surpassing state-of-theart fully supervised methods by approximately 4% in terms of mIOU.

![image](https://github.com/user-attachments/assets/c89ef9b2-aa07-4558-8ff0-e31b227f744d)

## Usage

### Requirements

```
 - python >= 3.8
 - pytorch >= 1.12.1
 - lightning
 - segmentation_models_pytorch
 - segment_anything
 - tensorboard
 - tensorboardX
```

Clone the repository locally:

```
git clone https://github.com/xjwu1024/WPS-SAM.git
```

### Data Preparation

Download and extract PartImageNet dataset from [here](https://huggingface.co/datasets/turkeyju/PartImageNet/blob/main/PartImageNet_Seg.zip) The directory structure is expected to be:

```
/path/to/PartImageNet/
  annotations/
    test/
      img1.json
      img1.png
      ...
      test.json
    test_whole/
      img1.png
      ...
      test.json
    train/
      img2.json
      img2.png
      ...
      train.json
    train_whole/
      img2.png
      ...
      train.json
    val/
      img3.json
      img3.png
      ...
      val.json
    val_whole/
      img3.png
      val.json
  images/
    test/
      img1.JPEG
      ...
    train/
      img2.JPEG
      ...
    val/
      img3.JPEG
      ...
```

More details about the datases can be found in [here](https://github.com/TACJu/PartImageNet?tab=readme-ov-file)

### Train

To train WPS-SAM on PartImageNet run:

```
python train.py
```

### Evaluation

```
python eval.py
```

## Citation

If you use PPT or this repository in your work, please cite:
```
@inproceedings{wu2024wps,
  title={WPS-SAM: Towards Weakly-Supervised Part Segmentation with Foundation Models},
  author={Wu, Xin-Jian and Zhang, Ruisong and Qin, Jie and Ma, Shijie and Liu, Cheng-Lin},
  booktitle={European Conference on Computer Vision},
  pages={314--333},
  year={2024},
  organization={Springer}
}
```
and provide a link to this repository as a footnote or a citation.

