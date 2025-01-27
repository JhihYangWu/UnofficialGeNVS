# Unofficial Implementation of [GeNVS](https://arxiv.org/abs/2304.02602)
Alex Berian, Daniel Brignac, JhihYang Wu, Natnael Daba, Abhijit Mahalanobis  
University of Arizona  
Built off of [EDM](https://github.com/NVlabs/edm) from Nvidia  
**Please consider citing our paper [CrossModalityDiffusion](https://arxiv.org/abs/2501.09838) if you found our code useful.**

CrossModalityDiffusion: Multi-Modal Novel View Synthesis with Unified Intermediate Representation  
Accepted to WACV 2025 GeoCV Workshop  
arXiv: https://arxiv.org/abs/2501.09838  
Code: https://github.com/alexberian/CrossModalityDiffusion  

Abstract: Geospatial imaging leverages data from diverse sensing modalities-such as EO, SAR, and LiDAR, ranging from ground-level drones to satellite views. These heterogeneous inputs offer significant opportunities for scene understanding but present challenges in interpreting geometry accurately, particularly in the absence of precise ground truth data. To address this, we propose CrossModalityDiffusion, a modular framework designed to generate images across different modalities and viewpoints without prior knowledge of scene geometry. CrossModalityDiffusion employs modality-specific encoders that take multiple input images and produce geometry-aware feature volumes that encode scene structure relative to their input camera positions. The space where the feature volumes are placed acts as a common ground for unifying input modalities. These feature volumes are overlapped and rendered into feature images from novel perspectives using volumetric rendering techniques. The rendered feature images are used as conditioning inputs for a modality-specific diffusion model, enabling the synthesis of novel images for the desired output modality. In this paper, we show that jointly training different modules ensures consistent geometric understanding across all modalities within the framework. We validate CrossModalityDiffusion's capabilities on the synthetic ShapeNet cars dataset, demonstrating its effectiveness in generating accurate and consistent novel views across multiple imaging modalities and perspectives.

---
### BibTeX
```
@misc{berian2025crossmodalitydiffusionmultimodalnovelview,
      title={CrossModalityDiffusion: Multi-Modal Novel View Synthesis with Unified Intermediate Representation}, 
      author={Alex Berian and Daniel Brignac and JhihYang Wu and Natnael Daba and Abhijit Mahalanobis},
      year={2025},
      eprint={2501.09838},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.09838}, 
}
```

---
### Files and Purpose
```
dnnlib/ -- unmodified code from EDM
torch_utils/ -- unmodified code from EDM
training/ -- modified code from EDM, where training loop is stored
genvs/ -- new code needed for GeNVS

inference/ -- where inference script is stored

train.py -- script to initiate training of GeNVS
train.sh -- to easily run train.py
```

---
### Environment Setup
```
conda env create -f environment.yml
conda activate genvs
```

---
### Downloading Data to Train On
`srn_cars.zip` from https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR  
Hosted by authors of pixelNeRF  

---
### Downloading our Trained Weights
Download https://drive.google.com/file/d/1ESsrKM99MK3wBPmgSNthaGNdPXUM1WCm/view?usp=sharing  
unzip such that `/training_runs/00000-uncond-ddpmpp-edm-gpus8-batch96-fp32/network-snapshot-004000.pkl` exists  

---
### Inference our Trained Model
```
cd inference/
./inference.sh
```
Customize options inside inference.sh to your preference  
if `--video` flag is present, inference.py generates video instead of novel views at target poses  

To improve image quality, increase the number of inferences in the denoising diffusion process with `--denoising_steps=30`

How to format input for inference:  
`inference/input/rgb` images to input into GeNVS  
`inference/input/pose` corresponding poses of those images  
`inference/input/target_pose` poses of novel views that you want GeNVS to generate  


---
### Training GeNVS
```
./train.sh
```
tensorboard logging stored inside `training_runs/run_name/`  
to launch web server for visualize training progress:  
```
tensorboard --logdir=training_runs/run_name/
```
if your training computer can't host web apps due to firewall, download the tensorboard log file(s) inside `training_runs/run_name/` and run tensorboard on your PC  

Training progress should look something like https://wandb.ai/natedaba/GeNVS/runs/6fq07yj0  
You might need to change `--seed` a few times in `train.sh` if it seems like GeNVS is not training at all  

To specify which GPUs on your system you'd like to use, adjust the environment variable `CUDA_VISIBLE_DEVICES` in the `train.sh` script as well as the option  `--nproc_per_node=X` according to the number of GPUs you want to use. 
Example: 
```
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --standalone \
          --nproc_per_node=7 train.py --whatever_options_you_want\
```
Select the batch size according to the amount of RAM your GPUs have and the number of GPUs. It must be divisible by the number of GPUs.  

---
### Warnings for Custom Dataset
Your custom dataset class in `genvs/data/your_dataset.py` should return  
`focal` as: normalized focal length (do necessary computation inside getitem method)  
`z_near` as: absolute `z_near` / normalized focal length  
`z_far` as: absolute `z_far` / normalized focal length  

Make sure to briefly skim the first few lines of any script for dataset specific constants such as:  
Radius of sphere at which the images were taken/to be inferenced  
`z_near` `z_far` values  
`coord_transform`  
