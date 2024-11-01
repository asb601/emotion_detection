# Emotion Detection Using FER and Deep Learning

This project leverages **Facial Emotion Recognition (FER)** to analyze emotions in images and videos. It utilizes a Convolutional Neural Network (CNN) for detecting seven emotions—**angry, disgust, fear, happy, neutral, sad, and surprise**—and provides a demonstration using the FER library and OpenCV for both image and video inputs.

---

## Project Overview

### Features
- **Emotion Detection in Images**: Detect emotions in static images using the FER library, displaying confidence scores for each detected emotion.
- **Real-Time Emotion Detection in Videos**: Process each frame in a video file to recognize emotions in real time, with the option to save annotated output.
- **Deep Learning Model**: A custom CNN model trained on the FER2013 dataset to recognize emotions in grayscale images.

---

## Requirements

Ensure you have the following installed:

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- FER library
- Matplotlib
- Seaborn
- Numpy
- Pandas

Install dependencies via:

```bash
pip install -r requirements.txt
