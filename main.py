import cv2
import numpy as np
from tensorflow import keras

# Load the trained model (ensure the correct path to the model file)
model = keras.models.load_model('model.h5')

class Lanes:
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

    def road_lines(self, frame):
        small_img = cv2.resize(frame, (160, 80))
        small_img = np.array(small_img)
        small_img = small_img[None, :, :, :]

        prediction = model.predict(small_img)[0] * 255
        self.recent_fit.append(prediction)

        if len(self.recent_fit) > 5:
            self.recent_fit = self.recent_fit[1:]

        self.avg_fit = np.mean(np.array([i for i in self.recent_fit]), axis=0)

        blanks = np.zeros_like(self.avg_fit).astype(np.uint8)
        lane_drawn = np.dstack((blanks, self.avg_fit, blanks))

        lane_image = cv2.resize(lane_drawn, (1280, 720))

        # Ensure both frame and lane_image have the same data type (e.g., np.uint8)
        frame = frame.astype(np.uint8)
        lane_image = lane_image.astype(np.uint8)

        result = cv2.addWeighted(frame, 1, lane_image, 1, 0)

        return result

lanes = Lanes()

# Update the path to the input video file
input_video_path = "lanes_test.mp4"

# Open the video file
cap = cv2.VideoCapture(input_video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = lanes.road_lines(frame)

    # Display the processed frame
    cv2.imshow("Processed Frame", processed_frame)

    # Press 'q' to quit the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
