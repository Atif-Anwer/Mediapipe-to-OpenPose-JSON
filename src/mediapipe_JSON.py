"""
Mediapipe pose extraction and exporting to OpenPose format but Mediapipe has 33 keypoints as output as compared to 25 from Openpose. The keypoints also have a different order.
The code below extracts the keypoints from images in a folder and exports them as an Openpose JSON format with 25 keypoints.
the JSON is compaitble with SMPLify-X for 3D shape extraction
- Uses Hydra.cc for config
- Supports only 1 person per frame (Mediapipe limitation)
- Supports multiple image extensions in folder (PNG, JPG, JPEG etc)

Requirements:
      Hydra for config
      Mediapipe

INPUT:          - Folder Path Containing RGB Images
RETURNS:        - Mediapipe outputkeypoints in OpenPose compatible JSON format
"""

import json
import logging
from glob import glob
from os.path import join
from pickle import FALSE

import cv2
import hydra
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from google.protobuf.json_format import MessageToDict
from matplotlib.animation import FuncAnimation
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf

# Configuring python logger
log = logging.getLogger(__name__)

# Adding hydra config file to load parameters at file load
@hydra.main(version_base = None, config_path = "conf", config_name = "config")
def main(cfg: DictConfig):
	"""Main function."""
	log.debug("Debug level message")

	# debug print for hydra
	print(OmegaConf.to_yaml(cfg))

	# ---- Mediapipe config ----
	mp_drawing        = mp.solutions.drawing_utils
	mp_holistic       = mp.solutions.holistic
	mp_pose           = mp.solutions.pose
	mp_drawing_styles = mp.solutions.drawing_styles

	# Defining the defauly Openpose JSON structure
	# (template from openpose output)
	json_data = {
		"version": 1.3,
		"people": [
			{
				"person_id"              : [-1],
				"pose_keypoints_2d"      : [],
				"face_keypoints_2d"      : [],
				"hand_left_keypoints_2d" : [],
				"hand_right_keypoints_2d": [],
				"pose_keypoints_3d"      : [],
				"face_keypoints_3d"      : [],
				"hand_left_keypoints_3d" : [],
				"hand_right_keypoints_3d": []
			}
		]
	}

	# Batch loading all images (path specified by the Hydra cfg file)
	img_folder = []
	for ext in ('*.png', '*.jpg', '*.jpeg'):
		img_folder.extend(glob(join(str(cfg.files.test_img_path), ext)))

	# Loop over all the sorted  video files in the folder
	for img in natsorted(img_folder):
		# Debug print file name
		log.info(f"Processing image file: {img}")

		count = 0
		with mp_holistic.Holistic(	min_detection_confidence=0.5,
						min_tracking_confidence=0.75,
						model_complexity=2,
						smooth_landmarks = True) as holistic:
			frame = cv2.imread(img)
			# getting image dimensions for scaling the x,y coordinates of the keypoints
			height, width, _ = frame.shape
			count += 1

			# ------------- MEDIAPIPE DETECTION IN FRAME -------------
			results = holistic.process(frame)
			landmarks = results.pose_landmarks.landmark

			if cfg.params.write_json == True:
				tmp =[]
				onlyList = []
				list4json = []

				# converting the landmarks to a list
				for idx, coords in enumerate(landmarks):
					coords_dict = MessageToDict(coords)
					# print(coords_dict)
					qq = (coords_dict['x'], coords_dict['y'], coords_dict['visibility'])
					tmp.append(qq)

				#  SCALING the x and y coordinates with the resolution of the image to get px corrdinates
				for i in range(len(tmp)):
					tmp[i] = ( int(np.multiply(tmp[i][0], width)), \
						   int(np.multiply(tmp[i][1], height)), \
						   tmp[i][2])
				# Calculate the two additional joints for openpose and add them
				# NECK KPT
				tmp[1] = ( (tmp[11][0] - tmp[12][0]) / 2 + tmp[12][0], \
					   (tmp[11][1] + tmp[12][1]) / 2 , \
					   0.95 )
				# saving the hip mid point in the list for later use
				stash = tmp[8]
				# HIP_MID
				tmp.append(stash)
				tmp[8] = ( (tmp[23][0] - tmp[24][0]) / 2 + tmp[24][0], \
					   (tmp[23][1] + tmp[24][1]) / 2 , \
					    0.95 )

				# Reordering list to comply to openpose format
				# For the order table,refer to the Notion page
				# restoring the saved hip mid point
				mp_to_op_reorder = [0, 1, 12, 14, 16, 11, 13, 15, 8, 24, 26, 28, 23, 25, 27, 5, 2, 33, 7, 31, 31, 29, 32, 32, 30, 0, 0, 0, 0, 0, 0, 0, 0]

				onlyList = [tmp[i] for i in mp_to_op_reorder]

				# delete the last 8 elements to conform to OpenPose joint length of 25
				del onlyList[-8:]

				# OpenPose format requires only a list of all landmarkpoints. So converting to a simple list
				for nums in onlyList:
					for val in nums:
						list4json.append(val)

				# Making the JSON openpose format and adding the data
				json_data = {
					"version": 1.3,
					"people": [
						{
							"person_id"              : [-1],
							"pose_keypoints_2d"      : list4json,
							"face_keypoints_2d"      : [],
							"hand_left_keypoints_2d" : [],
							"hand_right_keypoints_2d": [],
							"pose_keypoints_3d"      : [],
							"face_keypoints_3d"      : [],
							"hand_left_keypoints_3d" : [],
							"hand_right_keypoints_3d": []
						}
					]
				}

				json_filename = str(img) + ".json"
				json_filename = json_filename.replace(".png","_keypoints")
				with open(json_filename, 'w') as fl:
					fl.write(json.dumps(json_data, indent=2, separators=(',', ': ')))

				log.info(f"Writing JSON file: {json_filename}")
			# plt.close(fig)
		cv2.destroyAllWindows()



# ------------------------------------------------
if __name__ == "__main__":
	main()


# REFERENCES:
# 1. https://github.com/google/mediapipe/issues/1020
# 2. https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/871