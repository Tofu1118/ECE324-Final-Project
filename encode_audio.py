import os
import numpy as np
import librosa
from librosa import feature
import soundfile as sf
import torch
import torchvision
import autoencoder
from torch import nn
import ffmpeg

def save_to_csv(filename, arr):
    np.save(filename, arr)

def spectrogram_to_audio(spectrogram_vec, sr, filename):
    curr_dir = os.getcwd()
    os.chdir(os.getcwd() + "\\audio_output\\")
    audio = feature.inverse.mel_to_audio(spectrogram_vec, sr = sr)
    filename = filename.replace('.mp3', '.wav')
    sf.write(file = filename, data = audio, samplerate = sr)
    os.chdir(curr_dir)

if __name__ == "__main__":
    model = autoencoder.autoencoder_deep()
    model.load_state_dict(torch.load("autoencoder_deep_weights.pth"))

    dir = os.getcwd() + "\\audio_input\\"
    filenames = os.listdir(dir)
    for file in filenames:
        y, sr = librosa.load(dir + file, sr=44100)
        length = sr * 10
        total = len(y) // sr
        start = np.random.randint(total - 10)
        extract_arr = y[start:start + length:]
        arr_s = librosa.feature.melspectrogram(extract_arr, sr)
        spectrogram_to_audio(arr_s, sr, filename = file)
        output_s = model(torch.from_numpy(arr_s).unsqueeze(0).unsqueeze(0).float()).squeeze(0).squeeze(0).detach().numpy()
        spectrogram_to_audio(output_s, sr, filename = file.replace('.mp3', ' decoded.wav'))