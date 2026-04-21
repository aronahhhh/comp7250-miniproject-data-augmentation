# comp7250-miniproject-data-augmentation

COMP7250 course project: Data augmentation for image classification.

This project uses CIFAR-10 and a small CNN to compare several data augmentation settings. The main question is whether augmentation can improve test accuracy and reduce overfitting.

## Files

- `Experiment.py`: main experiment code for all data augmentation comparisons.
- `results/`: output folder for CSV files and figures.

## Install

```bash
pip install -r requirements.txt
```

## Run experiments

```bash
python Experiment.py
```

This command runs the full comparison experiment with four settings:

- `baseline`: no data augmentation
- `flip`: random horizontal flip
- `crop`: random crop
- `flip_crop`: random horizontal flip and random crop

The script uses Apple MPS for training on Mac.

The experiment uses 20 epochs, batch size 64, Adam optimizer, and learning rate 0.001.

## Outputs

After running `Experiment.py`, the following files are saved in `results/`:

- `experiment_results.csv`: train/test loss and accuracy for every epoch
- `test_accuracy.png`: test accuracy curves
- `loss_curves.png`: train and test loss curves
- `best_accuracy.png`: best test accuracy comparison

These plots can be used directly in the project report.

## Current Result Summary

In the current experiment, all augmentation settings improve the best test accuracy compared with the baseline. The baseline reaches a higher training accuracy but has a larger train-test gap, which indicates overfitting. The augmentation settings reduce this gap and give more stable test performance.

