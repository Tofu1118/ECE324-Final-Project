import torch
import torch.nn as nn
import torch.onnx
import os
import graphviz
import ensemble_cnn
import ensemble_resnet
import autoencoder
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    train = np.load("resnet_train.npy")
    val = np.load("resnet_val.npy")
    plt.title("Model 2 Training and Validation Accuracy over Epochs")
    plt.plot(train, label = "Training")
    plt.plot(val, label = "Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("Ensemble_Resnet_Accuracy.pdf")
    plt.show()

    print(train[-1])
    print(val[-1])

