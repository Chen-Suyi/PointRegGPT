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

![image text](./files/pipeline.png)

## Installation
Please use the following commands for installation.  
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

The 3DMatch data for **point cloud registration** can be downloaded from [OverlapPredator](https://github.com/prs-eth/OverlapPredator).

Please put it in `PointRegGPT/dataset/indoor/data`.
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
The training dataset for depth correction can be downloaded from [here](https://1drv.ms/u/c/072d47fe74bf5728/QShXv3T-Ry0ggAfLAAAAAAAAuzBw7zyZTQQk0w).

The unzipped data should be organized as follows:
```
--dataset--depth_correction--metadata--train.json
                          |         |--val.json
                          |--data--train--XXXXXX-input.depth.png
                                |      |--XXXXXX-label.depth.png
                                |--val--XXXXXX-input.depth.png
                                     |--XXXXXX-label.depth.png
```

### Official Generative Dataset
The generative dataset used in the experiments in our paper can be downloaded from [here](https://1drv.ms/f/c/072d47fe74bf5728/EihXv3T-Ry0ggAfMAAAAAAABsx_3AG5NpvTyCHXxyBfXSw).

The unzipped data should be organized as follows:
```
--generated_dataset--metadata--gt.log
                  |--data--scene-XXXXXX--sample-000000.cloud.ply
                                      |--sample-000001.cloud.ply
```

## Pre-trained Weights
We will provide pre-trained weights on the [releases](https://github.com/Chen-Suyi/PointRegGPT/releases) page.
- `successive_ddnm_diffusion_results.zip` contains the pre-trained weights for our diffusion model.<br>Please unzip and put it to `PointRegGPT/successive_ddnm_diffusion_results`.

- `depth_correction_results.zip` contains the pre-trained weights for our depth correction module.<br>Please unzip and put it to `PointRegGPT/depth_correction_results`.


## Quick Start
To begin with, please set the path to your downloaded 3DMatch RGB-D training data in [`generate_dataset.py#L48`](https://github.com/Chen-Suyi/PointRegGPT/blob/b65742a65321ab52848863fd070d21c32a13a157/generate_dataset.py#L48).

Use the following commands to create generative data:
```
# To generate point cloud pairs (No.000000 ~ No.000009 for example)
python generate_dataset.py --resume=official -start=0 -stop=10

# To generate a gt.log for the pairs
python generate_gt.py -start=0 -stop=10
```

By default, the generated data will be saved in `PointRegGPT/generated_dataset`.

## Diffusion Model
Please set the path to your downloaded 3DMatch RGB-D training data in [`train_successive_ddnm_diffusion.py#L28`](https://github.com/Chen-Suyi/PointRegGPT/blob/b65742a65321ab52848863fd070d21c32a13a157/train_successive_ddnm_diffusion.py#L28).

### Training
You can try training your own diffusion model using the following command:  
```
python train_successive_ddnm_diffusion.py
```

By default, the checkpoints will be saved in `PointRegGPT/successive_ddnm_diffusion_results` with names formatted in `model-SUFFIX.pt`.

### Multi-GPU Training
The trainer has been equipped with [🤗Accelerator](https://huggingface.co/docs/accelerate/package_reference/accelerator), please refer to the link if you are not yet familiar with it.

Multi-GPU training can be simply set up by using the following commands:
```
# First, configure the Accelerator on your own machine
accelerate config

# Then, launch your training
accelerate launch train_successive_ddnm_diffusion.py
```

### Testing
You can check the state of a diffusion model during training by:
```
# Replace SUFFIX with the suffix of checkpoint
python test_successive_ddnm_diffusion.py --resume=SUFFIX
```

By default, the results will be saved in `PointRegGPT/successive_ddnm_diffusion_samples`.

## Depth Correction

### Training
You can also try training your own depth correction module using the following command:  
```
python train_depth_correction.py
```

By default, the checkpoints and `train.log` will be saved in `PointRegGPT/depth_correction_results`.

### Testing
Please set the path to your downloaded 3DMatch RGB-D testing data in [`test_depth_correction.py#L15`](https://github.com/Chen-Suyi/PointRegGPT/blob/b65742a65321ab52848863fd070d21c32a13a157/test_depth_correction.py#L15).

You can qualitatively test a depth correction module on 3DMatch RGB-D test split using the following command:  
```
python test_depth_correction.py --resume=SUFFIX
```

By default, the results will be saved as GIFs in `PointRegGPT/depth_correction_samples`.

## Point Cloud Registration
To load generated data during training, we provide examples in `PointRegGPT/example_dataloader` for the baselines ([PREDATOR](https://github.com/prs-eth/OverlapPredator), [CoFiNet](https://github.com/haoyu94/Coarse-to-fine-correspondences), and [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)) we used in our paper.

We also provide pre-trained weights on the [releases](https://github.com/Chen-Suyi/PointRegGPT/releases) page, which are trained on our official generative dataset. You can simply load them using the official code of each baseline.

## Citation
Coming soon.

## Achnowledgement
[Denoising Diffusion Probabilistic Model, in Pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
