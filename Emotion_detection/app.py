from fer import FER
import cv2
import os

# Load the image
image_path = "../png-transparent-girl-open-mouth-happy-thumbnail.png"  # Replace with the correct path to your image
if not os.path.exists(image_path):
    print("Image path is incorrect or the image does not exist.")
else:
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print("Could not open or find the image.")
    else:
        # Initialize the FER emotion detector with MTCNN
        detector = FER(mtcnn=True)

        # Detect emotions in the image
        emotions = detector.top_emotion(image)

        # Print the detected emotion
        if emotions:
            print(f"Detected Emotion: {emotions[0]}, Confidence: {emotions[1]}")
        else:
            print("No emotion detected. Ensure the image has a clear, detectable face.")
