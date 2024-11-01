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

---

## Data Preparation

1. **Download the FER2013 dataset** from [Kaggle](https://www.kaggle.com) or another source.
2. **Store the dataset** in a designated directory, and ensure your data paths are correctly configured in the scripts.

---

## File Descriptions

- **Image Emotion Detection**: Code for analyzing static images to detect emotions and display confidence levels.
- **Video Emotion Detection**: Code for processing video frames to detect and annotate emotions in real time.
- **Model Training Script**: Script to train a custom CNN on the FER2013 dataset, using data augmentation and dropout for improved accuracy.
- **Confusion Matrix Plotting**: Script to generate a normalized confusion matrix for model evaluation.

---

## Getting Started

### 1. Image Emotion Detection
To detect emotions in an image, run the `image_emotion_detection.py` script.

Example usage:
```bash
python image_emotion_detection.py --image_path "../path_to_image.png"

### 2. Video Emotion Detection
To process a video file and output an annotated video with detected emotions, run the `video_emotion_detection.py` script.

Example usage:
```bash
python video_emotion_detection.py --video_path "../path_to_video.mp4"
### 3. Model Training
To train a custom CNN model on the FER2013 dataset, run the `train_emotion_model.py` script. This script will save checkpoints and output a final trained model in the designated directory.

Example usage:
```bash
python train_emotion_model.py
### 4. Confusion Matrix
To visualize model performance across each emotion class, run the `plot_confusion_matrix.py` script.

Example usage:
```bash
python plot_confusion_matrix.py
## Example Outputs

### Image Detection
Displays the detected emotions and their confidence scores on an image.

### Video Detection
Processes a video and overlays detected emotions on each frame in real-time.

### Model Training
Uses `train_emotion_model.py` to start model training. The trained model and checkpoints are saved in the designated directory.

