# Galaxy Classification - Physics Applications of AI Project

<details>
<summary>Show/Hide Table of Contents</summary>

[[_TOC_]]

</details>

## How to run

This was tested and appeared to work on the devcontainer provided in: 
```
{
    "image": "registry.gitlab.unige.ch/ai-in-physics-course/exercises/base:latest",
    "forwardPorts": [8888],
}
```
Before running the program, ensure that the directory `data/image_training_rev1/*` contains the ~61,000 images from the training image dataset. This dataset is split into training, validation and testing automatically so downloading the additional test dataset is not necessary. There are around 300 images already in the folder that were placed to ensure that the dev container executes the function during testing, but these are not sufficient to replicate the data seen in the report. The labels file is small, and so is already provided in the data folder.

To run the program, run `run.py`. After around 10-15 seconds as it loads the depedencies, it will prompt you with a list of configs to run the file. These are configurable and are found in the `configs/` directory. You can carefully modify the ones you want, although it is not recommended without thoroughly checking as it may break the program. Two short models are given that allow for quick testing (in binary classification using Ex.1 labels and one in regression using Ex. 2 labels) that use 3 epochs of training rather than 25. The others are the configs that were used for the report:

```
Please select a config file by typing its corresponding number (ex: 3):
1. 3Epoch_BinC_Gray_80_Aug.yaml 
2. 3Epoch_RegC_RGB_70_Aug.yaml 
3. BinC_Gray_80_Aug.yaml
4. BinC_Gray_80_Drop.yaml
5. BinC_Gray_80_Everything.yaml
6. BinC_Gray_80_Nothing.yaml
7. BinC_RGB_50_Aug.yaml
8. BinC_RGB_80_Aug.yaml
9. BinC_RGB_80_Drop.yaml
10. BinC_RGB_80_Everything.yaml
11. BinC_RGB_80_Nothing.yaml
12. RegC_RGB_0_Aug_AllClasses.yaml
13. RegC_RGB_50_Aug.yaml
14. RegC_RGB_70_Aug.yaml
15. RegC_RGB_70_Aug_BinCArch.yaml
16. RegC_RGB_70_Drop.yaml
17. RegC_RGB_70_Everything.yaml
18. RegC_RGB_70_Nothing.yaml
```

Once selected, the program will automatically run through the pipeline, and produce a folder with the model name and time in the `results` directory. You will be provided with metrics, a log, a pdf confusion matrix, some pdf graphs pertaining to the training evolution and a saved model.

(Note: I ran into an error on a non-dev container environemnt when the image file directory was saved with two backslashes (\\) instead of a forward slash (/), preventing it from being processed. If this occurs, simply modify lines 111 and 120 in `utils.py` and change "/" to "\\". Although in the docker environment this should not occur.)

## Overview

I have chosen to focus on the Galaxy Classification Task, namely the first two tasks:

#### Exercise 1
In the first exercise, we only consider the top-level question (3 answers) and samples where at least 80 % of participants gave the same answer.
We assumed this to be a certain classification and transformed them to one-hot encoded labels.
The task for this part of the exercise is to use the image to predict the corresponding label.

#### Exercise 2

In the second exercise, we consider the second layer of questions from the original kaggle challenge, i.e. Q2 and Q7.
Here, we do not use one-hot encoded labels, but the original floats that range between 0 and 1, thus making the classification problem a regression problem.
Here we can use all images in the dataset (no separate copy is produced).
You should try to make sure that the output of the classifier matches the hierarchical structure of the questions, e.g. the the summed values for Q2 equal the value for answer Q1.1.

I have focused on the following studies:

#### Studies
* Does the same architecture (with a different output layer) perform well for both binary classification and regression tasks?
* Can you use augmentations to improve the classification performance?
* How do different confidence levels in the data impact model performance?
* Does color impact model performance, especially when considering the correlation between spirally-ness and the blue-ness of a galaxy?


## Data Used

You can download the image datasets from [here](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data) and the labels from [here](https://drive.switch.ch/index.php/s/UDMdgAxeYLSCzyU).
