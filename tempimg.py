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

        lane_image = cv2.resize(lane_drawn, (frame.shape[1], frame.shape[0]))

        # Ensure both frame and lane_image have the same data type (e.g., np.uint8)
        frame = frame.astype(np.uint8)
        lane_image = lane_image.astype(np.uint8)

        result = cv2.addWeighted(frame, 1, lane_image, 1, 0)

        return result

lanes = Lanes()

# Update the path to the input image file
input_image_path = "lane1.jpg"

# Read the input image
input_image = cv2.imread(input_image_path)

# Check if the input image was loaded successfully
if input_image is None:
    print("Error: Could not load the input image.")
else:
    # Perform lane detection on the input image
    processed_image = lanes.road_lines(input_image)
    # Display the processed image
    cv2.imshow("Processed Image", processed_image)

    # Wait for a key press and then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
