import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import ensemble_cnn
import ensemble_resnet
import torch
import os


def plot_conf_matrix(y_actual, y_pred, title):
    sns.set(font_scale = 0.6)
    cf_matrix = confusion_matrix(y_actual, y_pred)

    ax = sns.heatmap(cf_matrix, annot=True, fmt='', cmap='Blues')

    ax.set_title(title);
    ax.set_xlabel('\nPredicted Composer')
    ax.set_ylabel('Actual Composer');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(["Bach", "Vivaldi", "Mozart", "Beethoven", "Debussy", "Chopin", "Brahms", "Tchaikovsky"])
    ax.yaxis.set_ticklabels(["Bach", "Vivaldi", "Mozart", "Beethoven", "Debussy", "Chopin", "Brahms", "Tchaikovsky"])

    ## Display the visualization of the Confusion Matrix.
    plt.savefig(title + ".pdf")
    plt.show()


if __name__ == "__main__":
    x1_test = np.load(os.getcwd() + "\\data\\test\\spectrogram.npy")
    x2_test = np.load(os.getcwd() + "\\data\\test\\feature.npy")
    y_test = np.load(os.getcwd() + "\\data\\test\\label.npy")
    np.random.seed(45)
    test_index_list = np.arange(x1_test.shape[0])
    np.random.shuffle(test_index_list)
    x1_test = x1_test[test_index_list[0:480]]
    x2_test = x2_test[test_index_list[0:480]]
    y_test = y_test[test_index_list[0:480]]

    x1_test = torch.from_numpy(x1_test).unsqueeze(1).float()
    x2_test = torch.from_numpy(x2_test).unsqueeze(1).float()
    y_test = torch.tensor(y_test).type(torch.LongTensor)

    model = ensemble_resnet.Ensemble_Resnet()
    model.load_state_dict(torch.load('ensemble_resnet_weights.pth'))

    y_test_pred = model(x1_test, x2_test)
    y_test_pred = y_test_pred.argmax(axis=1)
    plot_conf_matrix(y_test, y_test_pred, 'Ensemble_Resnet')

    accuracy = (y_test_pred == y_test).sum()/len(x1_test)
    print(len(x1_test))
    print("The model has an accuracy of", accuracy*100, "on the test set.")