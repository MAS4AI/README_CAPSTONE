# Speech Aware

Speech Aware is a project focused on exploring the field of National Security by analyzing voice breakdown. The project incorporates four models to address different aspects related to speech analysis and recognition. These models are speaker recognition, speech enhancement, speech emotion recognition, and lip reading.

## Problem Statement

In the realm of National Security, understanding and interpreting speech plays a crucial role in various scenarios. However, there are several challenges associated with speech analysis, such as speaker recognition, noise interference, emotion recognition, and understanding speech from visual cues. The aim of the Speech Aware project is to develop models that can effectively address these challenges and provide valuable insights for National Security applications.

## Dataset

The Speech Aware project utilizes the following datasets:

1. **Our Self-Recorded Dataset**:
   - This dataset was recorded specifically for lips reading and speaker recognition tasks.
   - It consists of audio and video recordings of various individuals speaking.
   - The dataset includes corresponding labels for lips reading and speaker identities.

2. **VoxCeleb Dataset**:
   - The VoxCeleb dataset is used for training and evaluation in the speech enhancement model.
   - It is a large-scale dataset containing audio recordings from various celebrities.
   - The dataset provides clean speech signals for training the speech enhancement model.

3. **Video Dataset with Labeled Emotions**:
   - This dataset is employed for training and evaluation in the speech emotion recognition model.
   - It comprises video recordings of individuals expressing different emotions while speaking.
   - The dataset includes labeled emotions corresponding to each video.

## Objectives

The main objectives of the Speech Aware project are as follows:

1. **Speaker Recognition**: The speaker recognition model aims to identify and verify individuals based on their speech characteristics. It utilizes the SpeechBrain toolkit, which incorporates deep learning techniques, such as convolutional neural networks (CNNs) and recurrent neural networks (RNNs), to extract x-vectors. These x-vectors capture speaker-specific information, enabling tasks like enrollment, scoring, and classification for speaker verification or identification.

2. **Speech Enhancement**: The speech enhancement model focuses on improving the quality and intelligibility of speech signals corrupted by noise, reverberation, or other distortions. By leveraging the SpeechBrain toolkit, it employs CNNs and RNNs to train on labeled datasets of noisy and clean speech signals. The trained model can then suppress unwanted background noise in real-time, enhancing the speech signal.

3. **Speech Emotion Recognition**: This model utilizes Convolutional Neural Networks (CNNs) to extract relevant features from spectrograms or other time-frequency representations of the speech signal. Speech emotion recognition aims to analyze speech patterns and identify emotions expressed in the speech. The model can contribute to understanding the emotional content of speech for National Security applications.

4. **Lip Reading**: The lip reading model employs AV-HuBERT, a lip reading model based on the SpeechBrain toolkit. It focuses on understanding speech from visual cues by analyzing lip movements and extracting relevant information. Lip reading can assist in scenarios where audio may be unavailable or compromised, providing an alternative means of speech analysis for National Security purposes.

## Categories for the Four Models

1. **Speaker Recognition**:
   - Utilizes the SpeechBrain toolkit for speaker recognition tasks.
   - Extracts x-vectors using deep learning techniques like CNNs and RNNs.
   - Provides modules for speaker verification, identification, enrollment, scoring, and classification.
  
     # Speech Emotion Recognition Model

This is a deep learning model for speech emotion recognition using the RAVDESS Emotional Speech Audio Dataset.

## Dependencies

The model requires the following libraries:

- Matplotlib
- Librosa
- OS
- SciPy
- NumPy
- FastAI
- glob
- FetchLabel (a local class you need to define)

The code also uses the `sounddevice` library to record live audio for predictions.

## Process

1. The audio files are loaded using the Librosa library, which is used for music and audio analysis. Each file is converted into a Mel spectrogram, a graphical representation of the spectrum of frequencies of sound as they vary with time.

2. These spectrograms are then converted into decibel units for better audio signal representation.

3. The model extracts features from each audio file, identifies the expressed emotion, and saves the file in a designated folder according to the emotion.

4. A convolutional neural network (CNN) learner from the FastAI library is used to train the model on the spectrogram images. The learner uses a pre-trained ResNet-34 model and is trained for ten epochs.

5. The performance of the model is evaluated using a confusion matrix. The model can also be used to classify live audio recorded via the `sounddevice` library.


2. **Speech Enhancement**:
   - Implements speech enhancement techniques to improve speech signal quality.
   - Relies on the SpeechBrain toolkit for speech enhancement tasks.
   - Trains CNNs and RNNs on labeled datasets of noisy and clean speech signals.
   - Real-time suppression of background noise during speech enhancement.

3. **Speech Emotion Recognition**:
   - Leverages Convolutional Neural Networks (CNNs) for feature extraction.
   - Analyzes spectrograms or time-frequency representations of speech signals.
   - Identifies emotions expressed in speech for emotional content analysis.

4. **Lip Reading**:
   - Utilizes AV-HuBERT, a lip reading model based on the SpeechBrain toolkit.
   - Focuses on understanding speech from visual cues, specifically lip movements.
   - Extracts relevant information from lip movements for speech analysis.
     
     # AV-Hubert Video Classifier

This script is designed to download, process, and classify video data. The algorithm leverages deep learning techniques, specifically temporal convolutional networks (TCNs), to recognize and classify video clips. Here is a detailed step-by-step breakdown of the code.

## Preliminaries

The initial block of code navigates to the correct directory and clones the AV-Hubert repository from Facebook Research's GitHub. It initializes and updates any submodules within that repository. This is followed by the installation of various python packages like scipy, sentencepiece, python_speech_features, and scikit-video, which will be used later in the code.

## Download an Example Video

The next block takes our necessary files for video preprocessing including a shape predictor, mean face landmarks, and a sample video from an online source. It extracts the region of interest (ROI) of the video, which in this case is the mouth of the person speaking.

## Import a Pre-Trained Model

This block downloads a pre-trained model checkpoint and performs inference using the model. The inference generates a hypothesis on the contents of the video, specifically predicting the words that are spoken in the video.

## Inference Process

The inference process involves predicting the spoken words from the mouth ROI. It extracts features from the frames and uses the pre-trained model to generate an output hypothesis. The hypothesis is later split into individual words.

## Positive and Negative Word Lists

A list of "good" words and "bad" words are created. These lists will be used to classify the output of the model.

## Word Embedding and Clustering

The Sentence Transformers library is used to convert the words into vectors in a high-dimensional space. Then, UMAP (Uniform Manifold Approximation and Projection) is used to reduce the dimension of the vectors to 2D. Afterward, K-means clustering is applied to classify the vectors into two clusters, representing "good" and "bad" words.

## Logistic Regression

It trains a logistic regression model using the 2D word embeddings and the labels obtained from K-means clustering. This model can then be used to predict whether a new word is "good" or "bad".

## Visual Feature Extraction

Lastly, the algorithm extracts visual features from the video using the pre-trained model. It normalizes the frames, applies a center crop, and converts the frames into a tensor which is processed by the pre-trained model to obtain the final feature vector.
