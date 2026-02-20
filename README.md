# scPerturb-eval

This repository contains the pipeline developed to systematically assess model performance and evaluation metrics in single-cell perturbation studies.
The pipeline consists of three main steps:

1. Training
2. Prediction
3. Evaluation

---

## Getting started

Clone the repository using:

```bash
git clone https://github.com/scPerturbationStudies/scPerturb-eval.git
cd scPerturb-eval/scproj
```

---

## Configuration

The pipeline is driven by configuration files that define both datasets and models.

### Dataset configuration

Each single cell perturbation dataset is associated with a configuration file that specifies the **modality**, defined as the variable whose perturbation effect is to be predicted (e.g.  IFN-β stimulation), as well as the **primary**, which corresponds to the target cell type for which the perturbation effect is evaluated.

### Model configuration

Model configuration files define the model architecture, the associated hyperparameters and all training-related settings used during model optimization.

---

## Data Availability

Processed datasets can be added to the same folder. For each new dataset, a corresponding configuration file should be created and placed in the `./configs/data` folder.

---

## Training

Models are trained under two generalization settings:

* **Out-of-Distribution (OOD)**
  Training data include perturbed and unperturbed cells from all cell types *except* the target cell type.
  For the target cell type, only unperturbed cells are available during training, while all perturbed cells are excluded.

* **Partially In-Distribution (PID)**
  A fraction of perturbed cells from the target cell type (e.g. 20%) is included in the training set, while the remaining perturbed cells are held out.

Example training command:

```bash
python training.py \
  --models noperturb scPRAM \
  --datasets kang \
  --config_path ./configs \
  --save_folder ood \
  --setting ood
```

---

## Prediction

After training, models are used to predict perturbed profiles of the target cell type.

In both OOD and PID settings, the trained model is provided with **unperturbed cells of the target cell type** and tasked with predicting their corresponding perturbed states.

Example prediction command:

```bash
python prediction.py \
  --models noperturb scPRAM \
  --datasets kang \
  --config_path ./configs \
  --save_folder ood \
  --setting ood
```

---

## Evaluation

The evaluation step uses the **CrossSplit framework** to systematically assess model performance across different evaluation metrics.
The pipeline also includes **custom evaluation functions** designed to examine and analyze specific evaluation metrics (e.g. noise_effect_mixing_index)

Example evaluation command:

```bash
python evaluation.py \
  --models noperturb scPRAM \
  --datasets kang \
  --config_path ./configs \
  --base_path ../outputs \
  --setting ood \
  --evaluation_metric rigorous_wasserstein rigorous_mixing_index_seurat
```

---
