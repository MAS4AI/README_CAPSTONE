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

Please provide the necessary details for the datasets to be used in the project.
