# [ECCV 2024] PointRegGPT: Boosting 3D Point Cloud Registration using Generative Point-Cloud Pairs for Training
This is the Pytorch implementation of our ECCV2024 paper [PointRegGPT](https://arxiv.org/abs/2407.14054).

## Introduction
Data plays a crucial role in training learning-based methods for 3D point cloud registration.
However, the real-world dataset is expensive to build, while rendering-based synthetic data suffers from domain gaps.
In this work, we present PointRegGPT, boosting 3D point cloud registration using generative point-cloud pairs for training.
Given a single depth map, we first apply a random camera motion to re-project it into a target depth map. Converting them to point clouds gives a training pair.
To enhance the data realism, we formulate a generative model as a depth inpainting diffusion to process the target depth map with the re-projected source depth map as the condition.
Also, we design a depth correction module to alleviate artifacts caused by point penetration during the re-projection.
To our knowledge, this is the first generative approach that explores realistic data generation for indoor point cloud registration.
When equipped with our approach, several recent algorithms can improve their performance significantly and achieve SOTA consistently on two common benchmarks.

## Installation
Please use the following command for installation.  
```
# Download the codes
git clone https://github.com/Chen-Suyi/PointRegGPT.git
cd PointRegGPT

# It is recommended to create a new environment
conda create -n pointreggpt python==3.9
conda activate pointreggpt

# Install packages
pip install -r requirements.txt
```

## Data Preparation
### 3DMatch
The 3DMatch **RGB-D** dataset can be found on [3DMatch: RGB-D Reconstruction Datasets](https://3dmatch.cs.princeton.edu/).

Please follow the official [training and testing scenes split](http://vision.princeton.edu/projects/2016/3DMatch/downloads/rgbd-datasets/split.txt) to organize the downloaded files as follows:
```
--3DMatch--train--7-scenes-chess--camera-intrinsics.txt
        |      |               |--seq-XX--frame-XXXXXX.color.png
        |      |                       |--frame-XXXXXX.depth.png
        |      |                       |--frame-XXXXXX.pose.txt
        |      |--...
        |--test--7-scenes-redkitchen--camera-intrinsics.txt
              |                    |--seq-XX--frame-XXXXXX.color.png
              |                            |--frame-XXXXXX.depth.png
              |                            |--frame-XXXXXX.pose.txt
              |--... 
```

The 3DMatch data for **point cloud registration** can be downloaded from [OverlapPredator](https://github.com/prs-eth/OverlapPredator) by running:
```
wget --no-check-certificate --show-progress https://share.phys.ethz.ch/~gsg/pairwise_reg/3dmatch.zip
```

Please unzip and put it in `PointRegGPT/dataset/indoor/data`
The data should be organized as follows:
```
--dataset--indoor--metadata
                |--data--train--7-scenes-chess--cloud_bin_0.pth
                      |      |               |--...
                      |      |--...
                      |--test--7-scenes-redkitchen--cloud_bin_0.pth
                            |                    |--...
                            |--...
```

### Depth Correction
The training dataset for depth correction can be downloaded from [here](https://1drv.ms/u/s!AihXv3T-Ry0HgUu7MHDvPJlNBCTT?e=wgYXgz).

### Official Generative Dataset
The generative dataset used in the experiments in our paper can be downloaded from [here](https://1drv.ms/f/s!AihXv3T-Ry0HgUx04nb-Ku-TozY8?e=kQLHaR).

## Quick Start
Coming soon.
