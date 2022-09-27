# Speech Emotion Recognition
A Speech Recognition system built with DL techniques. The objective is to build a system which can classify 8 different emotions from human voice recordings.

The emotions are:

- Anger
- Disgust
- Calm
- Happiness
- Sadness
- Neutral
- Fear
- Surprise


# The Dataset

It consists in 12162 .wav files, each containing a short phrase in English language, recorded from many different speakers from all around the world, with age ranging between 20s and 60s.

The dataset was built by merging 4 different datasets:

- RAVDESS (1440 recordings)
- SAVEE (480  recordings)
- CREMA-D (7442  recordings)
- TESS (2800 recordings)


Data was even distributed for each class, except for the 'Calm' class which only has 192 data points associated with it.

# Preprocessing

The typical audio classification approach consists in decoding .wav files in numpy arrays and apply STFT (Short-Time Fourier Transform) to get  a bi-dimensional tensor, representing the spectrogram of each audio file.

In this way the audio classification problem can be treated as a typical CV classification problem.

In this specific case however, MFCC (Mel-frequency cepstral coefficients) was used to represent each audio file. Before feeding it to the CNN, data was normalized.

# The Model

CNN (Convolutional Neural Network) was considered for this task.

_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 25, 216, 32)       320       
                                                                 
 batch_normalization (BatchN  (None, 25, 216, 32)      128       
 ormalization)                                                   
                                                                 
 max_pooling2d (MaxPooling2D  (None, 12, 108, 32)      0         
 )                                                               
                                                                 
 dropout (Dropout)           (None, 12, 108, 32)       0         
                                                                 
 conv2d_1 (Conv2D)           (None, 12, 108, 32)       9248      
                                                                 
 batch_normalization_1 (Batc  (None, 12, 108, 32)      128       
 hNormalization)                                                 
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 6, 54, 32)        0         
 2D)                                                             
                                                                 
 dropout_1 (Dropout)         (None, 6, 54, 32)         0         
                                                                 
 conv2d_2 (Conv2D)           (None, 6, 54, 32)         9248      
                                                                 
 batch_normalization_2 (Batc  (None, 6, 54, 32)        128       
 hNormalization)                                                 
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 3, 27, 32)        0         
 2D)                                                             
                                                                 
 dropout_2 (Dropout)         (None, 3, 27, 32)         0         
                                                                 
 conv2d_3 (Conv2D)           (None, 3, 27, 32)         9248      
                                                                 
 batch_normalization_3 (Batc  (None, 3, 27, 32)        128       
 hNormalization)                                                 
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 1, 13, 32)        0         
 2D)                                                             
                                                                 
 dropout_3 (Dropout)         (None, 1, 13, 32)         0         
                                                                 
 flatten (Flatten)           (None, 416)               0         
                                                                 
 dense (Dense)               (None, 64)                26688     
                                                                 
 dropout_4 (Dropout)         (None, 64)                0         
                                                                 
 batch_normalization_4 (Batc  (None, 64)               256       
 hNormalization)                                                 
                                                                 
 dropout_5 (Dropout)         (None, 64)                0         
                                                                 
 dense_1 (Dense)             (None, 8)                 520       
                                                                 
=================================================================
Total params: 56,040
Trainable params: 55,656
Non-trainable params: 384



# Performance

The train set is 9.729 files and the test set 2.433 files

| Test Performance (Accuracy) | Test Performance (Precision) | Test Performance (Recall) |
| ------------- | ------------- | ------------- |
| 69.5%  | 80.7 %  | 57.2%  |

