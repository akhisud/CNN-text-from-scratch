# CNN-text-from-scratch

This is the keras implementation of the CNN model given in the paper 'Text Understanding from Scratch' (https://arxiv.org/pdf/1502.01710.pdf).
Toy datasets are used currently, but the repo will be updated soon to use the original datasets used by the authors of the paper.

##Implementation:

Python3, keras API with tensorflow as the backend

##Run using: 

python3 -W ignore agara.py dataset_train.tsv dataset_test.tsv

##Files:

1) main.py - Main file, contains code for model implementation, training and testing
2) data.py - Auxillary file for dataset handling
3) dataset_train.tsv - TSV file with toy training data. The character sequences have been generated randomly and don't have any meaning.
4) dataset_test.tsv - TSV file with toy test data. The character sequences have been generated randomly and don't have any meaning.

In both the data files, the format used for each sample is: Character_sequence \TAB Class_label

##Notes:

1) All output to console
2) The program generates a file called 'model_structure_new.png' (I've already placed it in this folder), in which it prints the entire structure of the ConvNet
3) The program generates a .hdf5 file in which it saves the best model during training
