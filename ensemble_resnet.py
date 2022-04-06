import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from tqdm import tqdm
import os


# model definition
# Using CNNs
class Ensemble_Resnet(nn.Module):
    def __init__(self):
        # call super to initialize the class above in the hierarchy
        super(Ensemble_Resnet, self).__init__()

        # spectrogram pre-trained resnet

        self.network1 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained = True)
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=3, bias=False)
        #self.network1.conv1 = nn.Conv2d(1, 64, kernel_size = 3, stride = 2, padding = 3, bias = False)

        # feature CNN
        self.network2 = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU())

        # combined
        self.linear = nn.Linear(6136, 8)

    def forward(self, x1, x2):
        #one = self.network1.conv1(self.network1(x1))
        one = self.network1(self.conv1(x1))
        two = self.network2(x2)

        # flatten and concat the two matrices
        x = torch.cat((torch.flatten(one, start_dim=1), torch.flatten(two, start_dim=1)), dim=1)

        return torch.sigmoid(self.linear(x))


def plot(title, y, y_label):
    plt.title(title)
    plt.plot(y)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.show()


if __name__ == "__main__":
    np.random.seed(45)
    # load training, validation, and testing
    x1_train = np.load(os.getcwd() + "\\data\\train\\spectrogram.npy")
    x2_train = np.load(os.getcwd() + "\\data\\train\\feature.npy")
    y_train = np.load(os.getcwd() + "\\data\\train\\label.npy")

    x1_train = torch.from_numpy(x1_train).unsqueeze(1).float()
    x2_train = torch.from_numpy(x2_train).unsqueeze(1).float()
    y_train = torch.tensor(y_train).type(torch.LongTensor)

    x1_val = np.load(os.getcwd() + "\\data\\validate\\spectrogram.npy")
    x2_val = np.load(os.getcwd() + "\\data\\validate\\feature.npy")
    y_val = np.load(os.getcwd() + "\\data\\validate\\label.npy")

    x1_val = torch.from_numpy(x1_val).unsqueeze(1).float()
    x2_val = torch.from_numpy(x2_val).unsqueeze(1).float()
    y_val = torch.tensor(y_val).type(torch.LongTensor)

    # define parameters
    epochs = 20
    lr = 0.0001
    batch = 200
    # Initialize model
    model = Ensemble_Resnet()

    # Initialize loss function and optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay = 0.001)
    train_index_list = np.arange(x1_train.shape[0])

    # Training
    for epoch in range(epochs):
        start = 0
        accuracy = 0
        val_accuracy = 0
        #curr_loss = torch.empty(1)
        np.random.shuffle(train_index_list)

        while start < len(x1_train):
            # get a new training data
            curr_batch = train_index_list[start:start + batch]
            curr_x1_train = x1_train[curr_batch, :, :]
            curr_x2_train = x2_train[curr_batch, :, :]
            curr_y_train = torch.tensor(y_train[curr_batch]).type(torch.LongTensor)
            start += batch

            y_pred = model(curr_x1_train, curr_x2_train)
            accuracy += (y_pred.argmax(axis=1) == curr_y_train).sum()
            curr_loss = loss(y_pred, curr_y_train)
            curr_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        val_start = 0
        while val_start < len(x1_val):
            curr_x1_val = x1_val[val_start:val_start + batch, :, :]
            curr_x2_val = x2_val[val_start:val_start + batch, :, :]
            curr_y_val = torch.tensor(y_val[val_start:val_start + batch]).type(torch.LongTensor)
            val_start += batch

            y_pred_val = model(curr_x1_val, curr_x2_val)
            val_accuracy += (y_pred_val.argmax(axis=1) == curr_y_val).sum()

        print('epoch:', epoch + 1,  'training accuracy =', accuracy, ' ', accuracy / len(x1_train))
        print('validation accuracy =', val_accuracy, ' ', val_accuracy/len(x1_val))

    torch.save(model.state_dict(), 'ensemble_resnet_weights.pth')