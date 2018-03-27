# TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation

By [Vladimir Iglovikov](https://www.linkedin.com/in/iglovikov/) and [Alexey Shvets](https://www.linkedin.com/in/alexey-shvets-b0215263/)

# Introduction

TernausNet is a modification of the celebrated UNet architecture that is widely used for binary Image Segmentation. For more details, please refer to our [arXiv paper](https://arxiv.org/abs/1801.05746).

![UNet11](https://habrastorage.org/webt/hu/ji/ir/hujiirvpgpf7eswq88h_x7ahliw.png)

(Network architecure)

![loss_curve](https://habrastorage.org/webt/no/up/xq/noupxqqk_ivqwv3e7btyxtemt0m.png)

Pre-trained encoder speeds up convergence even on the datasets with a different semantic features. Above curve shows validation Jaccard Index (IOU) as a function of epochs for [Aerial Imagery](https://project.inria.fr/aerialimagelabeling/)

This architecture was a part of the [winning solutiuon](http://blog.kaggle.com/2017/12/22/carvana-image-masking-first-place-interview/) (1st out of 735 teams) in the [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge).

# Citing TernausNet
Please cite TernausNet in your publications if it helps your research:

```
@ARTICLE{arXiv:1801.05746,
         author = {V. Iglovikov and A. Shvets},
          title = {TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation},
        journal = {ArXiv e-prints},
         eprint = {1801.05746}, 
           year = 2018
        }
```

# Example of the train and test pipeline

https://github.com/ternaus/robot-surgery-segmentation
