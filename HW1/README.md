BIOS 740: Homework 1
Predicting Alzheimer's Disease Status Using Hippocampal Data
Cassi Chen

Please review the important notes at the bottom!

This repository contains code to classify Alzheimer’s Disease (AD) using hippocampal data. The project compares a custom CNN against a custom ResNet18 architecture. Analysis and relevance are included.

All files are in the same directory.

Raw Data:
    - LeftCSV_organized
    - RightCSV_organized
    - ADNIMERGE_01Oct2024.csv
Cleaned Data: 
    - samples_with_dx_manifest.tsv
Models:
    - hw1.ipynb
    - hw1.py (same as .ipynb but in .py format)
Images:
    - train_val_acc_SimpleCNN.png
    - train_val_acc_SimpleResNet.png
    - train_val_loss_SimpleCNN.png
    - train_val_loss_SimpleResNet.png 
    - sample.png
Results: 
    - slurm-29239987.out
    - hw1_SimpleCNN.csv
    - hw1_SimpleResNet.csv
Slurm:
    - hw1.sl

IMPORTANT NOTES:

All code was originally written in hw1.ipynb and then converted to hw1.py in order to use hw1.sl. Numerical results can be found in slurm-29239987.out, while all image outputs are in .png format. All written texts and anlysis are in hw1.ipynb. Predictions are in hw1_SimpleCNN.csv and hw1_SimpleResNet.csv. Please review hw1.ipynb, slurm-29239987.out, __.csv, and __.png for a complete picture of the project. 
