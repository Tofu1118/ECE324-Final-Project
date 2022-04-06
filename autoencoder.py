import os

import torch
import torchvision
from torch import nn
import numpy as np

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=1))


        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding='same'),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding='same'),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding='same'),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(num_features=1))


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x

class autoencoder_shallow(nn.Module):
    def __init__(self):
        super(autoencoder_shallow, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=1))


        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding='same'),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding='same'),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(num_features=1))


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x

class autoencoder_deep(nn.Module):
    def __init__(self):
        super(autoencoder_deep, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=1))


        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding='same'),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
            nn.UpsamplingBilinear2d(scale_factor=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding='same'),
            nn.UpsamplingBilinear2d(scale_factor=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding='same'),
            nn.UpsamplingBilinear2d(scale_factor=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding='same'),
            nn.UpsamplingBilinear2d(scale_factor=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding='same'),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(num_features=1))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x

if __name__ == "__main__":
    np.random.seed(45)
    # load training, validation, and testing
    x1_train = np.load(os.getcwd() + "\\data\\train\\spectrogram.npy")
    #x2_train = np.load(os.getcwd() + "\\data\\train\\feature.npy")
    y_train = np.load(os.getcwd() + "\\data\\train\\label.npy")

    x1_train = torch.from_numpy(x1_train).unsqueeze(1).float()
    #x2_train = torch.from_numpy(x2_train).unsqueeze(1).float()
    y_train = torch.tensor(y_train).type(torch.LongTensor)

    x1_val = np.load(os.getcwd() + "\\data\\validate\\spectrogram.npy")
    #x2_val = np.load(os.getcwd() + "\\data\\validate\\feature.npy")
    y_val = np.load(os.getcwd() + "\\data\\validate\\label.npy")

    x1_val = torch.from_numpy(x1_val).unsqueeze(1).float()
    #x2_val = torch.from_numpy(x2_val).unsqueeze(1).float()
    y_val = torch.tensor(y_val).type(torch.LongTensor)

    # define parameters
    epochs = 20
    lr = 0.001
    batch = 1

    # Initialize model
    model = autoencoder_deep()

    # Initialize loss function and optimizer
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    train_index_list = np.arange(x1_train.shape[0])

    # define training procedure
    for epoch in range(epochs):
        start = 0
        total_loss = torch.empty(1)
        total_val_loss = torch.empty(1)
        np.random.shuffle(train_index_list)
        while start < len(x1_train):
            # get a new training data
            #print("starting training new mini-batch")
            curr_batch = train_index_list[start:start + batch]
            curr_x1_train = x1_train[curr_batch, :, :]
            #curr_x1_train = curr_x1_train[:, :, 0:124, 0:856]
            start += batch

            x_reconstructed = model(curr_x1_train)
            #print(curr_x1_train.shape)
            #print(x_reconstructed.shape)
            curr_loss = loss(x_reconstructed, curr_x1_train[:, :, 0:120, 0:852])
            total_loss += curr_loss
            curr_loss.backward()
            #total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        val_start = 0
        while val_start < len(x1_val):
            curr_x1_val = x1_val[val_start:val_start+batch, :, :]
            #curr_x1_val = curr_x1_val[:, :, 0:124, 0:856]
            val_start += batch

            x_val_reconstructed = model(curr_x1_val)
            total_val_loss += loss(x_val_reconstructed, curr_x1_val[:, :, 0:120, 0:852])

        print('epoch:', epoch + 1,  'training loss =', total_loss.item()/len(x1_train))
        print('validation loss =', total_val_loss.item()/len(x1_val))

        #implement early stopping
        '''if total_loss.item()/len(x1_train) < 50 and total_loss.item()/len(x1_train) > 0 and total_val_loss.item()/len(x1_val) < 70 and total_val_loss.item()/len(x1_val) > 0:
            print('Loss values acceptable, training procedure is early stopped.')
            break'''

    torch.save(model.state_dict(), 'autoencoder_deep_weights.pth')