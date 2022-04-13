# ECE324_Music_classification

This is a repository containing the source code, result visualizations, and raw data for the music classification and 
compression project conducted for the summative project of ECE324 taught at the University of Toronto in Winter 2022.

**Composers.py** is a web-browsing script written using Selenium Webdriver in Python to automatically download available audio from
some 8 selected composers from imslp.org. Selected mp3 recordings downloaded using **Composers.py** are included in the
**raw** folder and serve as the raw data that will be preprocessed to produce the training/test/validation sets.

**data_convert.py** uses librosa to convert the raw mp3 files into melspectrogram and audio feature vectors. The
outputted vectors are saved as .npy files in a folder called **data**, these processed .npy files are too large to be uploaded onto GitHub under a 
free account. **data_visualizer.py** contains functions to visualize the compositions of the processed data, as well as
to convert a melspectrogram array into a spectrograph.

**ensemble_cnn.py** contains the ensemble cnn model and its training procedures.

**ensemble_resnet.py** contains the cnn + resnet model and its training procedures.

**result_visualizer.py** plots the training and validation accuracy over training epochs. The plots can be found in **Ensemble_CNN_Accuracy.pdf** and **Ensemble_Resnet_Accuracy.pdf**.

**autoencoder.py** contains three autoencoder models and training procedures. **encode_audio.py** is a script that takes
raw mp3 audio contained inside the **audio_input** folder, and outputs their autoencoder reconstructions in the **audio_output**
folder. Audio outputs for the regular, deep, and shallow autoencoder models in previous runs are saved in **audio_output_regular**, **audio_output_deep**,
and **audio_output_shallow** respectively.

**result_visualizer.py** is a training script that runs the test data through classification models and visualize their 
performance in confusion matrix outputs. **Ensemble_CNN.pdf** and **Ensemble_Resnet.pdf** contain such confusion matrix 
outputs. 
