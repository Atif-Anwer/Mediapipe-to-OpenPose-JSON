import json
from glob import glob
from os.path import join

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from google.protobuf.json_format import MessageToDict
from imutils.video import FPS, FileVideoStream
from tqdm import tqdm


def vid_to_JSON() -> None:
	# Load the video
	video_path = '/home/atif/Documents/Mediapipe_to_OpenPose_JSON/videos/test2.mp4'
	video_label = 'test2'
	# cap = cv2.VideoCapture(video_path)

	# Initialize MediaPipe
	# mp_drawing = mp.solutions.drawing_utils
	# mp_holistic = mp.solutions.holistic

	# Sauce: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python#video
	BaseOptions           = mp.tasks.BaseOptions
	PoseLandmarker        = mp.tasks.vision.PoseLandmarker
	PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
	VisionRunningMode     = mp.tasks.vision.RunningMode

	model_path = '/home/atif/Dropbox/Apeira/CODES/Pose_Extraction/pose_landmarker_heavy.task'
	options = PoseLandmarkerOptions(
			base_options              = BaseOptions(model_asset_path=model_path),
			running_mode              = VisionRunningMode.VIDEO,
			output_segmentation_masks = True)

	# ---- Video reader and writer ----
	fourcc = cv2.VideoWriter_fourcc(*'XVID') # type: ignore
	# writer = None

	# Batch loading all the videos in the target folder (path specified by the Hydra cfg file)
	count      = 0
	framecount = []
	tmp        = []
	onlyList   = []
	list4json  = []

	json_data = {
			"label":str(video_label),
			"label_index":0,
			"data":[]
			}

	generateFormat = 'SemGCN' # 'COCO17', 'Body25' or 'SemGCN'

	# Process each frame in the video
	with PoseLandmarker.create_from_options(options) as landmarker:
		fvs = FileVideoStream(video_path).start()
		width  		  = int(fvs.stream.get(cv2.CAP_PROP_FRAME_WIDTH))   # vide `width`
		height 		  = int(fvs.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))  # video `height
		total_frames      = int(fvs.stream.get(cv2.CAP_PROP_FRAME_COUNT))   # video frames

		skipped_frames    = 0
		fps               = FPS().start()
		axes_weights=[1.0, 1.0, 0.3, 1.0, 1.0]  # noqa: F841

		print(f"{total_frames=}")
		for count in tqdm(range(total_frames)):

			# Convert the frame to RGB
			frame      = cv2.cvtColor(fvs.read(), cv2.COLOR_BGR2RGB)
			# frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			# count     += 1
			fps.update()

			mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
			pose_landmarker_result = landmarker.detect_for_video(mp_image, count)
			landmarks = (pose_landmarker_result.pose_landmarks[0])


			# scale the z dimension for all landmarks by 0.2 and replace with the updated z value
			temp = [landmarks[i].z * axes_weights[2] for i in range(len(landmarks))]
			for i in range(len(landmarks)):
				landmarks[i].z = temp[i]

			framecount.append(count)
			pose = []
			score = []
			# converting the landmarks to a list
			for idx, coords in enumerate(landmarks):
				# coords_dict = MessageToDict(coords)
				# print(coords_dict)
				pose.append([coords.x, coords.y])
				score.append(round( coords.visibility, 5))
				# qq = (coords_dict['x'], coords_dict['y'], coords_dict['visibility'])
				# tmp.append(qq)

			#  SCALING the x and y coordinates with the resolution of the image to get px corrdinates
			for i in range(len(pose)):
				pose[i] = ( 	round( (np.multiply(pose[i][0], width)), 3), \
						round( (np.multiply(pose[i][1], height)), 3) )

			# ----------------- CALCULATING NEW KEPOINTS -----------------
			# NECK KPT
			# Calculate the two additional joints for openpose and add them
			# The y distance to be added (y_offset) is the lower y value of the two points
			y_offset = pose[11][1] if pose[11][1] <= pose[12][1] else pose[12][1]
			neck_kpt = ( 	(pose[11][0] - pose[12][0]) / 2 + pose[12][0], \
					(pose[11][1] + pose[12][1]) / 2 + y_offset)

			# PELVIS KPT
			# The y distance to be added (y_offset) is the lower y value of the two points
			y_offset = pose[23][1] if pose[23][1] <= pose[24][1] else pose[24][1]
			pelvis = ( 	(pose[23][0] - pose[24][0]) / 2 + pose[24][0],	\
					(pose[23][1] + pose[24][1]) / 2 + y_offset 	)

			# SPINE and THORAX KPT
			# for spine: x-axis is the same as pelvis, y-axis is the average of neck and pelvis

			x_offset = pelvis[0] if pelvis[0] <= neck_kpt[0] else neck_kpt[0]
			y_offset = pelvis[1] if pelvis[1] <= neck_kpt[1] else neck_kpt[1]
			spine 	 = ( 	(pelvis[0] - neck_kpt[0]) /2 + x_offset, \
	    				(pelvis[1] - neck_kpt[1]) /2 + y_offset  )

			thorax = neck_kpt	# TBD

			y_offset = pose[2][1] if pose[2][1] <= pose[5][1] else pose[5][1]
			head = ( 	(pose[2][0] - pose[5][0]) / 2 + pose[5][0], \
					(pose[2][1] + pose[5][1]) / 2 + y_offset)

			# ----------------- UPDATING THE KEYPOINTS -----------------

			if generateFormat == 'Body25':
				"""Generate JSON compatible with OpenPose (Body25) format
				(NOT WORKING atm)
				"""
				onlyList   = []
				# saving the hip mid point in the list for later use
				stash = pose[8]
				# HIP_MID
				pose.append(stash)
				pose[8] = ( (pose[23][0] - pose[24][0]) / 2 + pose[24][0], \
						(pose[23][1] + pose[24][1]) / 2 )

				# Reordering list to comply to openpose format
				# For the order table,refer to the Notion page
				# restoring the saved hip mid point
				mp_to_op_reorder = [0, 1, 12, 14, 16, 11, 13, 15, 8, 24, 26, 28, 23, 25, 27, 5, 2, 33, 7, 31, 31, 29, 32, 32, 30, 0, 0, 0, 0, 0, 0, 0, 0]

				onlyList = [pose[i] for i in mp_to_op_reorder]
				score    = [[i] for i in score]
				# delete the last 8 elements to conform to OpenPose joint length of 25
				del onlyList[-8:]
				del score[-8:]

			elif generateFormat == 'SemGCN':
				"""Generate JSON compatible with SemGCN (GAST-Net) format
				"""
				onlyList   = []
				semGCN_reorder_fromMP = [00, 23, 25, 27, 24, 26, 28, 00, 00, 00, 00, 2, 3, 4, 5, 6, 7]

				onlyList = [pose[i] for i in semGCN_reorder_fromMP]
				score = [score[i] for i in semGCN_reorder_fromMP]

				# replace the temp values with actual keypoints (that are not calculated in Mediapipe)
				onlyList[0]  = pelvis
				onlyList[7]  = spine
				onlyList[8]  = thorax
				onlyList[9]  = neck_kpt
				onlyList[10] = head

				# update the score value
				score    = [[i] for i in score]

			elif generateFormat == 'COCO17':
				"""Generate JSON compatible with OpenPose (Body25) format
				(NOT WORKING atm)
				"""
				onlyList   = []
				# coco17_reorder = [0, 1, 12, 14, 16, 11, 13, 15, 24, 26, 28, 23, 25, 27, 5, 2, 8, 7]
				# onlyList = [pose[i] for i in coco17_reorder]
				# score = [score[i] for i in coco17_reorder]

				coco17_order = [0, 1, 12, 14, 16, 11, 13, 15, 0, 8, 9, 10, 11, 12, 13, 0, 14, 15, 16, 17, 0, 0, 0, 0, 0]
				pose_17 = [onlyList[i] for i in coco17_order ]
				score_17 = [score[i] for i in coco17_order ]

				# delete the 8th and 17 onwards elements to conform to OpenPose joint length of 17
				del pose_17[8]
				del pose_17[17:]

				del score_17[8]
				del score_17[17:]

				onlyList = pose_17
				score    = score_17

			# ----------------- GENERATE JSON -----------------
			qq = {
				"frame_index":	count+1,
				"skeleton":[
						{
							"pose" : onlyList,
							"score": score,
							"bbox" : [0, 0, 0, 0]
						}
					]
				}
			json_data["data"].append(qq)

		json_filename = video_label + ".json"
		json_filename = json_filename.replace(".png","_keypoints")
		with open(json_filename, 'w') as fl:
			fl.write(json.dumps(json_data, indent=2, separators=(',', ': ')))

	fps.stop()
	print("Skipped Frames: {:.2f}".format(skipped_frames))
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


# ------------------------------------------------
if __name__ == "__main__":
	vid_to_JSON()
