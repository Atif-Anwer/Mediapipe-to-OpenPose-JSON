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
	video_path = '/home/atif/Documents/Mediapipe_to_OpenPose_JSON/videos/baseball.mp4'
	video_label = 'baseball'
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

			# ----------------- VISUALIZE 3D POSE -----------------
			# fig = plt.figure()
			# ax = fig.add_subplot(111, projection='3d')
			# qq = []
			# for idx, coords in enumerate(landmarks):
			# 	# coords_dict = MessageToDict(coords)
			# 	# print(coords_dict)
			# 	qq.append([coords.x, coords.y, coords.z])
			# #  SCALING the x and y coordinates with the resolution of the image to get px corrdinates
			# for i in range(len(qq)):
			# 	qq[i] = ( 	round( (np.multiply(qq[i][0], width)), 3),
			# 			round( (np.multiply(qq[i][1], height)), 3),
			# 			qq[i][2] )
			# pose3d = np.reshape(qq, (99,1))
			# pose3d = np.delete(pose3d, (19, 20, 21))
			# # channels: 96x1 vector. The pose to plot.
			# show3Dpose( pose3d, ax )
			# plt.show()

			# ----------------------------------
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
				pose[i] = ( 	round( (np.multiply(pose[i][0], width)), 3),
						round( (np.multiply(pose[i][1], height)), 3) )

			# ----------------- CALCULATING NEW KEPOINTS -----------------
			# NECK KPT
			# Calculate the two additional joints for openpose and add them
			# The y distance to be added (y_offset) is the lower y value of the two points
			y_offset = pose[11][1] if pose[11][1] <= pose[12][1] else pose[12][1]
			NECK = ( 	round( ((pose[11][0] - pose[12][0]) / 2 + pose[12][0]), 2),
					round( ((pose[11][1] - pose[12][1]) / 2 + y_offset), 2))

			# Hip_Mid KPT
			# The y distance to be added (y_offset) is the lower y value of the two points
			y_offset = pose[23][1] if pose[23][1] <= pose[24][1] else pose[24][1]
			HIP_MID = ( 	round(((pose[23][0] - pose[24][0]) / 2 + pose[24][0]), 2),
					round(((pose[23][1] - pose[24][1]) / 2 + y_offset), 2) 	)

			# SPINE and THORAX KPT
			# for spine: x-axis is the same as Hip_Mid, y-axis is the average of neck and Hip_Mid

			x_offset = NECK[0]
			y_offset = NECK[1]
			SPINE 	 = ( 	round(( (HIP_MID[0] - NECK[0]) /2 + x_offset), 2),
	    				round(( (HIP_MID[1] - NECK[1]) /2 + y_offset), 2)  )

			THORAX = NECK	# TBD

			y_offset = pose[2][1] if pose[2][1] <= pose[5][1] else pose[5][1]
			HEAD = ( 	round(( (pose[2][0] - pose[5][0]) / 2 + pose[5][0]), 2),
					round(( (pose[2][1] - pose[5][1]) / 2 + y_offset), 2))

			# ----------------- UPDATING THE KEYPOINTS -----------------

			if generateFormat == 'Body25':
				"""Generate JSON compatible with OpenPose (Body25) format
				(NOT WORKING atm)
				"""
				onlyList              = []
				body25_reorder_fromMP = [0, 00, 12, 14, 16, 11, 13, 15, 00, 24, 26, 28, 23, 25, 27, 5, 2, 8, 7, 31, 31, 29, 32, 32, 30]

				onlyList = [pose[i] for i in body25_reorder_fromMP]
				score_list = [score[i] for i in body25_reorder_fromMP]

				# replace the temp values with actual keypoints (that are not calculated in Mediapipe)
				onlyList[1]  = NECK
				onlyList[8]  = HIP_MID

				# update the score value
				score    = [[i] for i in score_list]

			elif generateFormat == 'SemGCN':
				"""Generate JSON compatible with SemGCN (GAST-Net) format
				"""
				onlyList              = []
				semGCN_reorder_fromMP = [00, 23, 25, 27, 24, 26, 28, 00, 00, 00, 00, 2, 3, 4, 5, 6, 7]

				onlyList   = [pose[i] for i in semGCN_reorder_fromMP]
				score_list = [score[i] for i in semGCN_reorder_fromMP]

				# replace the temp values with actual keypoints (that are not calculated in Mediapipe)
				onlyList[0]  = HIP_MID
				onlyList[7]  = SPINE
				onlyList[8]  = THORAX
				onlyList[9]  = NECK
				onlyList[10] = HEAD

				# update the score value
				score  = [[i] for i in score_list]

			elif generateFormat == 'COCO17':
				"""Generate JSON compatible with OpenPose (Body25) format
				(NOT WORKING atm)
				"""
				onlyList            = []
				coco17_order_fromMP = [00, 00, 12, 14, 16, 11, 13, 15, 24, 26, 28, 23, 25, 27, 5, 2, 8, 7, 00, 00, 00, 00, 00, 00, 00]

				onlyList   = [pose[i] for i in coco17_order_fromMP ]
				score_list = [score[i] for i in coco17_order_fromMP ]

				# replace the temp values with actual keypoints (that are not calculated in Mediapipe)
				onlyList[0] = HEAD
				onlyList[1] = NECK

				# update the score value
				score    = [[i] for i in score_list]

			# ----------------- GENERATE JSON -----------------
			qq = {
				"frame_index":count+1,
				"skeleton":[
						{
							"pose" :onlyList,
							"score":score,
							"bbox" :[200, 500, 100, 100]
						}
					]
				}
			json_data["data"].append(qq)

		json_filename = video_label + ".json"
		json_filename = json_filename.replace(".png","_keypoints")
		with open(json_filename, 'w') as fl:
			fl.write(json.dumps(json_data, indent=2, separators=(',', ':')))

	fps.stop()
	print("Skipped Frames: {:.2f}".format(skipped_frames))
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


def show3Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False): # blue, orange
	"""
	Visualize a 3d skeleton
	Source: https://github.com/una-dinosauria/3d-pose-baseline/issues/30

	Args
		channels: 96x1 vector. The pose to plot.
		ax: matplotlib 3d axis to draw on
		lcolor: color for left part of the body
		rcolor: color for right part of the body
		add_labels: whether to add coordinate labels
	Returns
		Nothing. Draws on ax.
	"""

	H36M_NAMES = ['']*32
	H36M_NAMES[0]  = 'Hip'
	H36M_NAMES[1]  = 'RHip'
	H36M_NAMES[2]  = 'RKnee'
	H36M_NAMES[3]  = 'RFoot'
	H36M_NAMES[4]  = 'RFootTip'
	H36M_NAMES[6]  = 'LHip'
	H36M_NAMES[7]  = 'LKnee'
	H36M_NAMES[8]  = 'LFoot'
	H36M_NAMES[12] = 'Spine'
	H36M_NAMES[13] = 'Thorax'
	H36M_NAMES[14] = 'Neck/Nose'
	H36M_NAMES[15] = 'Head'
	H36M_NAMES[17] = 'LShoulder'
	H36M_NAMES[18] = 'LElbow'
	H36M_NAMES[19] = 'LWrist'
	H36M_NAMES[25] = 'RShoulder'
	H36M_NAMES[26] = 'RElbow'
	H36M_NAMES[27] = 'RWrist'

	assert channels.size == len(H36M_NAMES)*3, "channels should have 96 entries, it has %d instead" % channels.size
	vals = np.reshape( channels, (len(H36M_NAMES), -1) )

	I   = np.array([1,2,3,4,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1 # start points
	J   = np.array([2,3,4,5,7,8,9,13,14,15,16,18,19,20,26,27,28])-1 # end points
	LR  = np.array([1,1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

	# Make connection matrix
	for i in np.arange( len(I) ):
		x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
		ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)
		ax.text(x[1], y[1], z[1], H36M_NAMES[J[i]] )

	# RADIUS = 750 # space around the subject
	RADIUS = 750 # space around the subject
	xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
	ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
	ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
	ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

	if add_labels:
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.set_zlabel("z")

	# Get rid of the ticks and tick labels
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_zticks([])

	ax.get_xaxis().set_ticklabels([])
	ax.get_yaxis().set_ticklabels([])
	ax.set_zticklabels([])
	ax.set_aspect('equal')

	# Get rid of the panes (actually, make them white)
	white = (1.0, 1.0, 1.0, 0.0)
	ax.w_xaxis.set_pane_color(white)
	ax.w_yaxis.set_pane_color(white)
	# Keep z pane

	# Get rid of the lines in 3d
	ax.w_xaxis.line.set_color(white)
	ax.w_yaxis.line.set_color(white)
	ax.w_zaxis.line.set_color(white)

# ------------------------------------------------
if __name__ == "__main__":
	vid_to_JSON()
