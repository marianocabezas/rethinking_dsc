# Rethinking the loss function for extremely unbalanced datasets using 3D patches

Project to test some toy examples on the DSC loss for highly unbalanced datasets.

Since the introduction of the so-called Dice loss function in 2016 by Miletari et al. [[1](1)], we have been using it in the medical imaging domain for highly unbalanced datasets to improve the results. Empirically, that seems to be the case, but can we actually prove it analitically? How is it related to the generalised Dice loss by Sudre et al. [[2](2)]? Is there any better option? How does it compare to the focal loss? This repository is an extension of a paper based on that to get empirical results on several datasets with a high imbalance.


[[1](https://arxiv.org/abs/1606.04797)] Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi, "**V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation**" *International Conference on 3D Imaging, Modeling, Processing, Visualization and Transmission (3DIMPVT)*, 2016.

[[2](https://link.springer.com/chapter/10.1007/978-3-319-67558-9_28)] Carole H. Sudre, Wenqi Li, Tom Vercauteren, Sebastien Ourselin, M. Jorge Cardoso, "**Generalised Dice Overlap as a Deep Learning Loss Function for Highly Unbalanced Segmentations**" *International Workshop on Deep Learning in Medical Image Analysis (DLMIA) - MICCAI*, 2017.
