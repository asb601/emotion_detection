import cv2
from fer import FER

# Initialize the FER detector
detector = FER(mtcnn=True)  # Use MTCNN for better accuracy

# Load the video file
video_path = "../woman2.mp4"  # Replace with your video file path
video_capture = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not video_capture.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the width and height of the video frames
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object to save the video
output_filename = 'emotion_detected_output.mp4'  # Name of the output video file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
video_writer = cv2.VideoWriter(output_filename, fourcc, 30.0, (frame_width, frame_height))

# Process each frame of the video
while True:
    # Read a new frame from the video
    ret, frame = video_capture.read()
    
    # Break the loop if there are no frames left
    if not ret:
        break
    
    # Detect emotions in the current frame
    emotions = detector.detect_emotions(frame)
    
    # Draw rectangles and labels for detected faces and emotions
    for emotion in emotions:
        box = emotion['box']  # The bounding box
        emotions_scores = emotion['emotions']  # The emotion scores
        
        # Get the top emotion
        top_emotion, top_score = max(emotions_scores.items(), key=lambda item: item[1])
        
        # Draw the rectangle around the face
        cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
        
        # Put the emotion label and confidence score on the frame
        label = f"{top_emotion}: {top_score:.2f}"
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Write the processed frame to the video file
    video_writer.write(frame)

    # Display the processed frame with detected emotions
    cv2.imshow('Video Emotion Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and the VideoWriter
video_capture.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Processed video saved as {output_filename}")
