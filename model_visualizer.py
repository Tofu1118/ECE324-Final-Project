import torch
import torch.nn as nn
import torch.onnx
import os
import graphviz
import ensemble_cnn
import ensemble_resnet
import autoencoder
import numpy as np

if __name__ == '__main__':
    '''x1_train = np.load(os.getcwd() + "\\data\\train\\spectrogram.npy")
    x2_train = np.load(os.getcwd() + "\\data\\train\\feature.npy")
    y_train = np.load(os.getcwd() + "\\data\\train\\label.npy")

    x1_train = torch.from_numpy(x1_train).unsqueeze(1).float()
    x2_train = torch.from_numpy(x2_train).unsqueeze(1).float()
    y_train = torch.tensor(y_train).type(torch.LongTensor)'''

    resnet = ensemble_resnet.Ensemble_Resnet()
    cnn = ensemble_cnn.CNN_Ensemble()
    deep_autoencoder = autoencoder.autoencoder_deep()
    shallow_autoencoder = autoencoder.autoencoder_shallow()
    regular_autoencoder = autoencoder.autoencoder()

    print(cnn)

    '''resnet.eval()
    torch.onnx.export(resnet, (x1_train[0:1], x2_train[0:1]), "ensemble_resnet.onnx", export_params=True, opset_version=10)

    cnn.eval()
    torch.onnx.export(cnn, (x1_train[0:1], x2_train[0:1]), "ensemble_cnn.onnx", export_params=True, opset_version=10)

    deep_autoencoder.eval()
    torch.onnx.export(deep_autoencoder, x1_train[0:1], "deep_autoencoder.onnx", export_params=True, opset_version=10)

    shallow_autoencoder.eval()
    torch.onnx.export(shallow_autoencoder, x1_train[0:1], "shallow_autoencoder.onnx", export_params=True, opset_version=10)

    regular_autoencoder.eval()
    torch.onnx.export(regular_autoencoder, x1_train[0:1], "regular_autoencoder.onnx", export_params=True, opset_version=10)
'''

