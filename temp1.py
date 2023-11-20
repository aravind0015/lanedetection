# import cv2
# import numpy as np
# from scipy.misc import imresize
# from moviepy.editor import VideoFileClip
# from tensorflow import keras

# # Load the trained model
# model = keras.models.load_model(r'model.hs')

# class Lanes:
#     def __init__(self):
#         self.recent_fit = []
#         self.avg_fit = []

#     def road_lines(self, image):
#         small_img = imresize(image, (80, 160, 3))
#         small_img = np.array(small_img)
#         small_img = small_img[None, :, :, :]

#         prediction = model.predict(small_img)[0] * 255
#         Lanes.recent_fit.append(prediction)

#         if len(Lanes.recent_fit) > 5:
#             Lanes.recent_fit = Lanes.recent_fit[1:]

#         Lanes.avg_fit = np.mean(np.array([i for i in Lanes.recent_fit]), axis=0)

#         blanks = np.zeros_like(Lanes.avg_fit).astype(np.uint8)
#         lane_drawn = np.dstack((blanks, Lanes.avg_fit, blanks))

#         lane_image = imresize(lane_drawn, (720, 1280, 3))

#         result = cv2.addWeighted(image, 1, lane_image, 1, 0)

#         return result

# lanes = Lanes()

# vid_input = VideoFileClip("lanes_clip.mp4")
# vid_output = 'lanes_clip_out_2.mp4'

# vid_clip = vid_input.fl_image(lanes.road_lines)
# vid_clip.write_videofile(vid_output, codec='libx264')
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from tensorflow import keras

# Load the trained model (ensure the correct path to the model file)
model = keras.models.load_model('model.h5')

class Lanes:
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

    def road_lines(self, image):
        small_img = cv2.resize(image, (160, 80))  # Resize using cv2.resize
        small_img = np.array(small_img)
        small_img = small_img[None, :, :, :]

        prediction = model.predict(small_img)[0] * 255
        self.recent_fit.append(prediction)

        if len(self.recent_fit) > 5:
            self.recent_fit = self.recent_fit[1:]

        self.avg_fit = np.mean(np.array([i for i in self.recent_fit]), axis=0)

        blanks = np.zeros_like(self.avg_fit).astype(np.uint8)
        lane_drawn = np.dstack((blanks, self.avg_fit, blanks))

        lane_image = cv2.resize(lane_drawn, (1280, 720))  # Resize using cv2.resize

        # Convert both arrays to np.uint8 before blending
        image = image.astype(np.uint8)
        lane_image = lane_image.astype(np.uint8)

        result = cv2.addWeighted(image, 1, lane_image, 1, 0)

        return result

lanes = Lanes()

# Update the path to the video file
vid_input = VideoFileClip("lanes_clip.mp4")
vid_output = 'lanes_clip_out_2.mp4'

vid_clip = vid_input.fl_image(lanes.road_lines)
vid_clip.write_videofile(vid_output, codec='libx264')
