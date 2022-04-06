import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
import os


# model definition
# Using CNNs
class CNN_Ensemble(nn.Module):
    def __init__(self):
        # call super to initialize the class above in the hierarchy
        super(CNN_Ensemble, self).__init__()

        # spectrogram CNN
        self.network1 = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU())

        # feature CNN
        self.network2 = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU())

        # combined
        self.linear1 = torch.nn.Sequential(nn.Linear(424, 16), nn.ReLU())
        self.linear2 = torch.nn.Sequential(nn.Linear(1272, 16), nn.ReLU())
        self.final = nn.Linear(32, 8)

    def forward(self, x1, x2):
        one = self.network1(x1)
        two = self.network2(x2)
        one = self.linear1(torch.flatten(one, start_dim = 1))
        two = self.linear2(torch.flatten(two, start_dim = 1))

        # flatten and concat the two matrices
        x = torch.cat((one, two), dim=1)

        return torch.sigmoid(self.final(x))


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
    lr = 0.001
    batch = 64

    # Initialize model
    model = CNN_Ensemble()

    # Initialize loss function and optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay = 1e-6)

    train_index_list = np.arange(x1_train.shape[0])

    # Training
    for epoch in range(epochs):
        start = 0
        accuracy = 0
        val_accuracy = 0
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
            curr_x1_val = x1_val[val_start:val_start+batch, :, :]
            curr_x2_val = x2_val[val_start:val_start+batch, :, :]
            curr_y_val = torch.tensor(y_val[val_start:val_start+batch]).type(torch.LongTensor)
            val_start += batch

            y_val_pred = model(curr_x1_val, curr_x2_val)
            val_accuracy += (y_val_pred.argmax(axis=1) == curr_y_val).sum()

        print('epoch:', epoch + 1,  'accuracy =', accuracy, ' ', accuracy / len(x1_train))
        print('validation accuracy =', val_accuracy, ' ', val_accuracy / len(x1_val))

        if val_accuracy/len(x1_val) > 0.5:
            print("Early breaking")
            break

    torch.save(model.state_dict(), 'ensemble_cnn_weights.pth')