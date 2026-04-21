# Data Augmentation for Image Classification Tasks

Name: `<CHEN Junlin>`  
Student ID: `<25400991>`  
Course: COMP7250

## 1. Introduction

Image classification is a basic task in computer vision. The input is an image, and the model needs to predict which class the image belongs to. In this project, I use a convolutional neural network (CNN) for image classification on the CIFAR-10 dataset.

One common problem in image classification is overfitting. A model may perform well on the training set, but not as well on the test set. This means the model has learned patterns that are too specific to the training images. Data augmentation is one way to reduce this problem. It creates random variations of the training images, such as flipped or cropped images, so the model can see more diverse examples during training.

The goal of this project is to test whether simple data augmentation methods can improve test accuracy and reduce overfitting.

## 2. Problem Definition

The main question of this project is:

> Does data augmentation help a CNN generalize better on an image classification task?

The image classification task is based on CIFAR-10. CIFAR-10 has 60,000 color images with size 32x32. There are 10 classes, such as airplane, car, bird, cat, deer, dog, frog, horse, ship, and truck. The dataset has 50,000 training images and 10,000 test images.

I compare a baseline model with three data augmentation settings:

| Experiment | Training Data Augmentation |
|---|---|
| baseline | No data augmentation |
| flip | Random horizontal flip |
| crop | Random crop with padding |
| flip_crop | Random horizontal flip and random crop |

The baseline is used as the reference result. The other three experiments are used to check whether augmentation improves test performance and reduces the train-test gap.

## 3. Methodology

### 3.1 Dataset and Preprocessing

The dataset used in this project is CIFAR-10. Since the images are RGB images, each image has three channels. I normalize the images using the commonly used CIFAR-10 mean and standard deviation:

```text
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)
```

The same normalization is used for both the training set and the test set. Data augmentation is only used on the training set. I do not apply random augmentation to the test set, because the test set should be fixed for a fair comparison.

### 3.2 Model

I use a small CNN model for all experiments. The purpose is to keep the model simple and focus on the effect of data augmentation.

The model has three convolutional layers:

- 3 input channels to 16 channels, followed by ReLU and max pooling
- 16 channels to 32 channels, followed by ReLU and max pooling
- 32 channels to 64 channels, followed by ReLU and max pooling

After the convolutional layers, the model uses:

- one fully connected layer with 128 hidden units
- ReLU
- dropout with probability 0.25
- one final fully connected layer with 10 outputs

The output size is 10 because CIFAR-10 has 10 classes.

### 3.3 Data Augmentation

I test four settings in total:

1. **baseline**: only `ToTensor` and normalization are used.
2. **flip**: random horizontal flip is added.
3. **crop**: random crop with padding 4 is added.
4. **flip_crop**: both random horizontal flip and random crop are used.



### 3.4 Training Setup

The implementation is written in PyTorch. I use cross-entropy loss for classification and Adam as the optimizer.

The training settings are:

| Setting | Value |
|---|---:|
| Epochs | 20 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss function | CrossEntropyLoss |


For each experiment, I initialize a new model and train it from scratch. After every epoch, I evaluate the model on the test set. The training and test metrics are saved in `results/experiment_results.csv`.

## 4. Experimental Result

### 4.1 Test Accuracy

The test accuracy curves are shown below.

![Test Accuracy](../results/test_accuracy.png)

All three augmentation settings achieve better best test accuracy than the baseline. The best result is from `flip`, with a best test accuracy of 0.7679. The `crop` and `flip_crop` experiments are close, with best test accuracy values of 0.7654 and 0.7640.

The improvement is not extremely large, but it is consistent. The baseline reaches a best test accuracy of 0.7449, while all augmentation settings are above 0.7640.

### 4.2 Loss Curves

The loss curves are shown below.

![Loss Curves](../results/loss_curves.png)

The baseline has the lowest training loss at the end, but its test loss becomes much higher. This is a sign of overfitting. The model fits the training set better and better, but the test loss does not improve in the same way.

The augmentation experiments have higher training loss, which is expected because the training images are randomly changed. However, their test loss is lower than the baseline. This suggests that augmentation makes training harder but helps the model generalize better.

### 4.3 Numerical Results

The best test accuracy comparison is shown below.

![Best Test Accuracy](../results/best_accuracy.png)

The detailed results are:

| Experiment | Final Train Acc | Final Test Acc | Best Test Acc | Best Epoch | Final Train Loss | Final Test Loss | Train-Test Gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.8762 | 0.7406 | 0.7449 | 15 | 0.3367 | 1.0019 | 0.1356 |
| flip | 0.8135 | 0.7635 | 0.7679 | 15 | 0.5280 | 0.7329 | 0.0500 |
| crop | 0.7510 | 0.7624 | 0.7654 | 18 | 0.7061 | 0.6889 | -0.0114 |
| flip_crop | 0.7345 | 0.7640 | 0.7640 | 20 | 0.7654 | 0.6778 | -0.0295 |

The baseline has the highest final training accuracy, 0.8762, but its final test accuracy is only 0.7406. The train-test gap is 0.1356, which is much larger than the gaps of the augmentation experiments.

For `flip`, the final train-test gap is 0.0500. For `crop` and `flip_crop`, the final test accuracy is even slightly higher than the final training accuracy. This can happen because augmented training images are harder than the original test images.

From these results, the main observation is that augmentation reduces overfitting. The gain in accuracy is moderate, but the train-test gap and loss curves show a clearer improvement in generalization.

## 5. Conclusion

In this project, I tested the effect of data augmentation on CIFAR-10 image classification. I trained the same CNN model under four settings: baseline, random horizontal flip, random crop, and flip plus crop.

The results show that data augmentation improves the best test accuracy compared with the baseline. The improvement is around 2 percentage points. More importantly, augmentation reduces overfitting. The baseline has high training accuracy but a much larger train-test gap. The augmented models have lower training accuracy, but better or more stable test accuracy.

Among the tested methods, random horizontal flip gives the highest best test accuracy. Random crop and flip_crop also improve over the baseline. However, the differences between augmentation methods are small. This means that augmentation is useful, but adding more augmentation does not always give a much better result.

One limitation of this project is that each experiment is run once with a fixed random seed. If there is more time, I would repeat the experiments with several random seeds and report the average accuracy.

Overall, the experiment shows that simple data augmentation can reduce overfitting and improve generalization for image classification.

