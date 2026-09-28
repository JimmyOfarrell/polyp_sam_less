<div align="center">    
 
# SAM-Guided Prompt Learning for Multiple Sclerosis Lesion Segmentation
F. Proietto Salanitri, G. Bellitto, S.Calcagno, U.Bagci, C. Spampinato, M.Pennisi

[![prl](https://img.shields.io/badge/PRLetters_2025-_-orange?style=flat-square)](https://www.sciencedirect.com/science/article/pii/S0167865525003708)



</div>

# 🔍 Overview

The official PyTorch implementation for __"SAM-Guided Prompt Learning for Multiple Sclerosis Lesion Segmentation"__. 

MS-SAMLess is a training-time distillation framework for Multiple Sclerosis lesion segmentation that leverages SAM only during training to learn dense task-specific prompts. At inference, SAM is fully removed and replaced with a lightweight aggregator that transforms the learned prompts into segmentation masks, eliminating manual prompting and drastically reducing computational cost. The method achieves state-of-the-art performance on MSLesSeg while remaining compact, fast, and easily deployable.

<div align="center">
 <img  title="method overview" alt="Method" src="./imgs/Method_overview.png" style="width: 99%;"/>
</div>

## 📄 Paper abstract

Accurate segmentation of Multiple Sclerosis (MS) lesions remains a critical challenge in medical image analysis due to their small size, irregular shape, and sparse distribution. Despite recent progress in vision foundation models—such as SAM and its medical variant MedSAM—these models have not yet been explored in the context of MS lesion segmentation. Moreover, their reliance on manually crafted prompts and high inference-time computational cost limits their applicability in clinical workflows, especially in resource-constrained environments.
In this work, we introduce a novel training-time framework for effective and efficient MS lesion segmentation. Our method leverages SAM solely during training to guide a prompt learner that automatically discovers task-specific embeddings. At inference, SAM is replaced by a lightweight convolutional aggregator that maps the learned embeddings directly into segmentation masks—enabling fully automated, low-cost deployment.
We show that our approach significantly outperforms existing specialized methods on the public MSLesSeg dataset, establishing new performance benchmarks in a domain where foundation models had not previously been applied. To assess generalizability, we also evaluate our method on pancreas and prostate segmentation tasks, where it achieves competitive accuracy while requiring an order of magnitude fewer parameters and computational resources compared to SAM-based pipelines.
By eliminating the need for foundation models at inference time, our framework enables efficient segmentation without sacrificing accuracy.
This design bridges the gap between large-scale pretraining and real-world clinical deployment, offering a scalable and practical solution for MS lesion segmentation and beyond.

<div align="center">
  <img src="./imgs/results.png" style="max-width: 90%"/>
</div>


# 📂 Repository details

This repo relies on the [AutoSAM Framework](https://github.com/talshaharabany/AutoSAM) which has been modified to include the features described in the **MS-SAMLess** paper. 

# 📦 Pretrained Models

This project relies on the Segment Anything Model (SAM) during **training only**.

Please download the official pretrained SAM weights from: https://github.com/facebookresearch/segment-anything

After downloading, place the checkpoint file in:

```text
cp\
```

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/MS-SAMLess/MS-SAMLess.git
cd MS-SAMLess
conda create -n ms-samless python=3.8
conda activate ms-samless
pip install -r requirements.txt
```
### 🖥️ Tested Environment

- **Python**: 3.8.19  
- **PyTorch**: 1.13.1  
- **CUDA**: 11.7  
- **GPU**: NVIDIA H100 80GB HBM3  

> Other PyTorch/CUDA combinations may work but have not been extensively tested.

# Usage

## Training

The general command to launch a training is:

## 🔹Phase 1
```shell
python main.py --epoches 200 -bs 8 -task mslesseg --exp_name example --split_path data/mslesseg_data_split_official.json --data_dir data/ -lr 0.01 --pos_rand_crop 0.85 --neg_rand_crop 0.15 --optim SGD --theashold_discretize 0.5 --image_key flair --out_channels 2 --order 85 --criterion dice_ce
```

## 🔹Phase 2
```shell
python main.py --epoches 100 -bs 8 -task mslesseg --exp_name phase2_example --split_path data/mslesseg_data_split_official.json --data_dir data/ --theashold_discretize 0.5 --use_sam False --use_standard_net True --Idim 512 --net_segmentor ConvUpsample --optim SGD -lr 0.01 --out_channels 2 --image_key flair --ckpt_path results/{path_example_phase1}/net_best.pth --n_fold 0 --segmentor_finetune_backbone True --criterion dice_ce
```
## Evaluation

## 🔹Phase 1

```shell
python test.py -task mslesseg --exp_name test_example --split_path data/mslesseg_data_split_official.json --data_dir data/ --theashold_discretize 0.5 --ckpt_path results/..../net_best.pth --image_key flair --out_channels 2
```

## 🔹Phase 2
```shell
python test.py -task mslesseg --exp_name test_phase2_example --split_path data/mslesseg_data_split_official.json --data_dir data/ --theashold_discretize 0.5 --use_sam False --use_standard_net True --Idim 512 --net_segmentor ConvUpsample --out_channels 2 --ckpt_path results/{path_phase2_example}/net_best.pth --image_key flair --net HardNetSegmentor
```

# 📊 Datasets and Data Splits

This repository does **not** distribute datasets nor predefined data split files.

Users are required to:
- Download the datasets independently, according to the original licenses.
- Create a JSON file defining the train/validation/test split.
- Place the split file inside the `data/` directory.
- Provide its path using the `--split_path` argument.

### 📁 Split File Structure

The split file must be a JSON file with the following structure:

- One or more folds (e.g. `fold0`, `fold1`, ...)
- Each fold contains a `train` and a `val` list
- A global `test` list
- Each sample is defined by a dictionary mapping modality names to file paths

```json
{
  "foldX": {
    "train": [
      {
        "<modality_1>": "<path_to_image>",
        "<modality_2>": "<path_to_image>",
        "...": "...",
        "mask": "<path_to_segmentation_mask>"
      }
    ],
    "val": [
      {
        "<modality_1>": "<path_to_image>",
        "<modality_2>": "<path_to_image>",
        "...": "...",
        "mask": "<path_to_segmentation_mask>"
      }
    ]
  },
  "test": [
    {
      "<modality_1>": "<path_to_image>",
      "<modality_2>": "<path_to_image>",
      "...": "...",
      "mask": "<path_to_segmentation_mask>"
    }
  ]
}
```

#### 🔎 Notes

- Modality keys (e.g. `flair`, `t1`, `t2`) must match the arguments provided at runtime (e.g. `--image_key flair`).
- File paths can be absolute or relative to the project root.
- The `mask` field is mandatory for all samples.
- Multiple folds can be defined for cross-validation by adding `fold1`, `fold2`, etc.



# 📜 Citation

```bibtex
@article{PROIETTOSALANITRI2026205,
title = {SAM-guided prompt learning for Multiple Sclerosis lesion segmentation},
journal = {Pattern Recognition Letters},
volume = {199},
pages = {205-211},
year = {2026},
issn = {0167-8655},
doi = {https://doi.org/10.1016/j.patrec.2025.11.018},
author = {Federica {Proietto Salanitri} and Giovanni Bellitto and Salvatore Calcagno and Ulas Bagci and Concetto Spampinato and Manuela Pennisi},
keywords = {Multiple Sclerosis, Segment Anything Model (SAM), Dense Prompt Learner, Brain MRI}
}
```
