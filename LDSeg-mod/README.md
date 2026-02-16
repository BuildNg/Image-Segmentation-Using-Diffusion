# LDSeg: Latent Diffusion for Ambiguous Segmentation

This repository contains the implementation of LDSeg, a model designed for medical image segmentation that captures inter-expert variability (ambiguity) using latent diffusion processes.

## Getting Started

### Environment Setup
It is recommended to use the provided `ldseg` conda environment:
```bash
conda activate ldseg
```

### Configuration
The model behavior and training process are controlled via two configuration files:

1.  **`model_config.ini`**: Defines the architecture of the LDSeg model, including the label encoder/decoder, image encoder, denoiser U-Net, and the latent distribution parameters.
    - *Key change*: `[NoiseScheduler] Timesteps` controls the number of diffusion steps.
2.  **`train_config.ini`**: Defines hyperparameters for the training loop (Learning Rate, Batch Size, Epochs) and data paths.
    - *Key change*: `[Data] DatasetDir` and `ValidationDir` must point to your LIDC dataset roots.
    - *Key change*: `[Device] Device` should be set to `cuda` for GPU training or `cpu` for local testing.

## Training

To start training the model, run:
```bash
python train.py
```
The script will initialize the model from `model_config.ini` and follow the parameters in `train_config.ini`. 

### Ambiguous Segmentation Metrics
During training, the model periodically (every 4 epochs) evaluates itself on the validation set using metrics designed for ambiguous segmentation:
- **GED (Generalized Energy Distance)**: Measures the distance between prediction and ground truth distributions.
- **Max Dice**: Measures the best possible overlap achieved by any model sample for each expert annotation.
- **CI (Collective Insight)**: A comprehensive metric that balances sensitivity, precision, and diversity.

Logs and model checkpoints will be saved in the directory specified by `[Logging] LogDir` in `train_config.ini`.

## Evaluation

### Pipeline Verification
To verify that your installation and configuration are correct, you can run the pipeline test suite:
```bash
python test_pipeline.py
```
This script performs a single training iteration (mocked) and a full metric computation step to ensure all components are working together correctly.

### Custom Evaluation
You can use the functions in `sampling.py` to run evaluation on custom data or checkpoints. The `compute_metrics_for_dataloader` function is the primary entry point for batch evaluation of ambiguous metrics.

## Metrics Details
- **Combined Sensitivity (Sc)**: Measures if the union of model predictions covers the union of expert opinions.
- **Diversity Agreement (Da)**: Measures if the level of disagreement among model samples matches the level of disagreement among experts.
- **Collective Insight (CI)**: The harmonic mean of $Sc$, $D_{max}$, and $Da$.

---
*Developed for CS89: Image Segmentation Using Diffusion*
