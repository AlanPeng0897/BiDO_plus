<h1 align="center"> BiDO+: Unknown-Aware Bilateral Dependency Optimization</h1>

This repository contains the implementation of BiDO+ for both low-resolution and high-resolution settings.

## Environment Setup

Create the environment from the provided YAML file:

```bash
conda env create -f environment.yml
conda activate bido_plus
```


## Project Structure

```
BiDO_plus/
├── high_resolution/  # Implementation for high-resolution tasks
│   ├── train_target.py
│   ├── compute_ood_score.py
│   └── ...
└── low_resolution/   # Implementation for low-resolution tasks
    ├── train_target.py
    ├── compute_ood_score.py
    └── ...
```

## Training

### Low Resolution Setting

Train a model on low-resolution images (64×64):

```bash
cd low_resolution
python train_target.py -c config/celeba.json \
    --enable_bido \
    --measure HSIC \
    --alpha 0.05 \
    --beta 0.5 \
    --enable_OE
```

### High Resolution Setting

Train a model on high-resolution images (224×224):

```bash
cd high_resolution
python train_target.py -c config/celeba.json \
    --enable_bido \
    --measure HSIC \
    --alpha 0.01 \
    --beta 0.1 \
    --enable_OE
```

### Arguments

- `--enable_bido`: Enable multi-layer BiDO constraints
- `--measure`: Select dependency measure (`HSIC` or `COCO`)
- `--alpha`: Weight for input-feature dependency 
- `--beta`: Weight for label-feature dependency 
- `--enable_OE`: Enable outlier exposure constraints
- `--strategy`: OOD detection strategy

## Evaluation

### OOD Detection

To evaluate OOD detection performance of the target model:

#### Low Resolution:
```bash
cd low_resolution
python compute_ood_score.py -c config/celeba.json --mode msp
```

#### High Resolution:
```bash
cd high_resolution 
python compute_ood_score.py -c config/celeba.json --mode msp
```

The script will compute and plot OOD detection scores, reporting metrics like AUROC and FPR.

### Model Inversion Attacks
To assess the robustness of trained models against potential privacy risks, we evaluate how well they can resist attempts to reconstruct private training data from their outputs. Specifically, we test the models under model inversion attacks such as [LOMMA](https://github.com/sutd-visual-computing-group/Re-thinking_MI) and [PPA](https://github.com/LukasStruppek/Plug-and-Play-Attacks), which aim to recover input samples using only model outputs and gradients.


## Citation

If you find this code useful for your research, please cite our paper:
```
@article{peng2025bido_plus,
  title   = {BiDO+: Unknown‐Aware Bilateral Dependency Optimization for Defending Against Model Inversion Attacks},
  author  = {Peng, Xiong and Liu, Feng and Wang, Nannan and Lan, Long and Liu, Tongliang and Cheung, Yiu‐ming and Han, Bo},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {99},
  pages   = {1--13},
  year    = {2025},
  publisher = {IEEE}
}
```

