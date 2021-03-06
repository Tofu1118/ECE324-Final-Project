# references https://gist.github.com/gchavez2/53148cdf7490ad62699385791816b1ea

import os
import numpy as np
import librosa
from librosa import feature
import ffmpeg

def get_feature_vector(y, sr):
    feature_vector = []
    for function in fn_list_1:
        r = function(y, sr)
    # if r is a 1d array
    if len(r.shape) == 1:
        r = r.unsqueeze(1)
    if not feature_vector:
        feature_vector = r
    else:
        feature_vector = np.concatenate((feature_vector, r), axis=0)
    for function in fn_list_2:
        feature_vector = np.concatenate((feature_vector, function(y)), axis=0)
    return feature_vector


def save_to_csv(filename, arr):
    np.save(filename, arr)


# returns label of the song
def get_label(filename):
    labels = {"Bach": [1, 0, 0, 0, 0, 0, 0, 0],
              "Vivaldi": [0, 1, 0, 0, 0, 0, 0, 0],
              "Mozart": [0, 0, 1, 0, 0, 0, 0, 0],
              "Beethoven": [0, 0, 0, 1, 0, 0, 0, 0],
              "Debussy": [0, 0, 0, 0, 1, 0, 0, 0],
              "Chopin": [0, 0, 0, 0, 0, 1, 0, 0],
              "Brahms": [0, 0, 0, 0, 0, 0, 1, 0],
              "Tchaikovsky": [0, 0, 0, 0, 0, 0, 0, 1]}
    for name in labels:
        if name in filename:
            return labels[name]
    return None

# returns label of the song
def get_label1(filename):
    labels = {"Bach": 0,
              "Vivaldi": 1,
              "Mozart": 2,
              "Beethoven": 3,
              "Debussy": 4,
              "Chopin": 5,
              "Brahms": 6,
              "Tchaikovsky": 7}
    for name in labels:
        if name in filename:
            return labels[name]
    return None

if __name__ == "__main__":
    dir = os.getcwd() + "\\raw\\"
    out_dir = os.getcwd() + "\\data\\"
    sub_dir = {"train\\", "validate\\", "test\\"}
    #sub_dir = {"train\\"}
    fn_list_1 = [
        feature.chroma_stft,
        feature.chroma_cqt,
        feature.chroma_cens,
        feature.mfcc,  #
        feature.spectral_centroid,  #
        feature.spectral_bandwidth,  #
        feature.spectral_contrast,  #
        feature.spectral_rolloff,  #
        feature.poly_features,
        feature.tonnetz,
        feature.tempogram  #
    ]
    fn_list_2 = [
        feature.rms,  #
        feature.spectral_flatness,
        feature.zero_crossing_rate  #
    ]

    # repeat this for train, validate, test
    for folder in sub_dir:
        feature_array = []
        spectrogram_array = []
        labels = []
        filenames = os.listdir(dir + folder)
        # open every file in the directory
        for file in filenames:
            label = get_label1(file)

            y, sr = librosa.load(dir+folder+file, sr=44100)
            # the length of a 10 sec clip
            length = sr*20
            sample_num = 0
            total = len(y)//sr
            if total < 20:
                continue

            # take 15 samples from each piece
            while sample_num < 15:
                start = np.random.randint(total-20)
                extract_arr = y[start*sr:start*sr+length:]
                # preprocess
                arr_f = get_feature_vector(extract_arr, sr)
                arr_s = librosa.feature.melspectrogram(extract_arr, sr)
                labels.append(label)
                # append the new arrays to the total array
                feature_array.append(arr_f)
                spectrogram_array.append(arr_s)

                sample_num += 1

        # convert to np arrays and save to file
        save_to_csv(out_dir + folder + "feature", feature_array)
        save_to_csv(out_dir + folder + "spectrogram", spectrogram_array)
        save_to_csv(out_dir + folder + "label", np.array(labels))

