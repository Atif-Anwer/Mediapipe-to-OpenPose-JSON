import cv2
import matplotlib.pyplot as plt
import mediapipe as mp

# Load the video
video_path = '/home/atif/Documents/Mediapipe_to_OpenPose_JSON/videos/test2.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Process each frame in the video
with mp_holistic.Holistic() as holistic:
        while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                        break

                # Convert the frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame with MediaPipe
                results = holistic.process(frame_rgb)

                # TODO: Perform further processing with the results

                # Display the annotated frame
                annotated_frame = frame.copy()
                mp_drawing.draw_landmarks(annotated_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                plt.imshow(annotated_frame)
                plt.pause(1e-15)

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
