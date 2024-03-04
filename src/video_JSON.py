import json
from enum import Enum

import cv2
import mediapipe as mp
import numpy as np
import rerun as rr
from imutils.video import FPS, FileVideoStream
from tqdm import tqdm

# from glob import glob
# from os.path import join
# import matplotlib.pyplot as plt
# from google.protobuf.json_format import MessageToDict
# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2


def vid_to_JSON() -> None:
	# Load the video
	video_path = '/home/atif/Documents/Mediapipe_to_OpenPose_JSON/videos/baseball.mp4'
	video_label = 'baseball'
	kpts = {	'NOSE'		  : 00 ,
		'LEFT_EYE_INNER'  : 1,
		'LEFT_EYE'        : 2,
		'LEFT_EYE_OUTER'  : 3,
		'RIGHT_EYE_INNER' : 4,
		'RIGHT_EYE'       : 5,
		'RIGHT_EYE_OUTER' : 6,
		'LEFT_EAR'        : 7,
		'RIGHT_EAR'       : 8,
		'MOUTH_LEFT'      : 9,
		'MOUTH_RIGHT'     : 10,
		'LEFT_SHOULDER'   : 11,
		'RIGHT_SHOULDER'  : 12,
		'LEFT_ELBOW'      : 13,
		'RIGHT_ELBOW'     : 14,
		'LEFT_WRIST'      : 15,
		'RIGHT_WRIST'     : 16,
		'LEFT_PINKY'      : 17,
		'RIGHT_PINKY'     : 18,
		'LEFT_INDEX'      : 19,
		'RIGHT_INDEX'     : 20,
		'LEFT_THUMB'      : 21,
		'RIGHT_THUMB'     : 22,
		'LEFT_HIP'        : 23,
		'RIGHT_HIP'       : 24,
		'LEFT_KNEE'       : 25,
		'RIGHT_KNEE'      : 26,
		'LEFT_ANKLE'      : 27,
		'RIGHT_ANKLE'     : 28,
		'LEFT_HEEL'       : 29,
		'RIGHT_HEEL'      : 30,
		'LEFT_FOOT_INDEX' : 31,
		'RIGHT_FOOT_INDEX': 32,
}

	use_rerun = False

	# Sauce: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python#video

	qq = frozenset({ (15, 21), (16, 20), (18, 20), (3, 7), (14, 16), (23, 25), (28, 30), (11, 23), (27, 31), (6, 8), (15, 17), (24, 26), (16, 22), (4, 5), (5, 6), (29, 31), (12, 24), (23, 24), (0, 1), (9, 10), (1, 2), (0, 4), (11, 13), (30, 32), (28, 32), (15, 19), (16, 18), (25, 27), (26, 28), (12, 14), (17, 19), (2, 3), (11, 12), (27, 29), (13, 15), })

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
	# fourcc = cv2.VideoWriter_fourcc(*'XVID') # type: ignore
	# writer = None

	# Batch loading all the videos in the target folder (path specified by the Hydra cfg file)
	count      = 0
	framecount = []
	onlyList   = []
	# tmp        = []
	# list4json  = []

	connections = Enum('connections', kpts)

	if use_rerun is True:
		DESCRIPTION = ""
		rr.init("ErgoPose", spawn=True)
		rr.log("description", rr.TextDocument(DESCRIPTION, media_type=rr.MediaType.MARKDOWN), timeless=True)
		rr.log(
			"/",
			rr.AnnotationContext(
			rr.ClassDescription(
				info=rr.AnnotationInfo(id=0, label="Person"),
				keypoint_annotations=[rr.AnnotationInfo(id=lm.value, label=lm.name) for lm in connections], # type: ignore
						keypoint_connections=qq, # type: ignore
			)
			),
			timeless=True,
		)
		rr.log( "video/mask",
			rr.AnnotationContext([
						rr.AnnotationInfo(id=0, label="Background"),
						rr.AnnotationInfo(id=1, label="Person", color=(0, 0, 0)),
						]),
			timeless=True,
			)
		rr.connect()

	json_data = {
			"label":str(video_label),
			"label_index":0,
			"data":[]
			}

	generateFormat = 'Original_COCO17' # 'COCO17', 'Body25' or 'Original_COCO17'

	# Process each frame in the video
	with PoseLandmarker.create_from_options(options) as landmarker:
		fvs = FileVideoStream(video_path).start()
		frame_width  		  = int(round(fvs.stream.get(cv2.CAP_PROP_FRAME_WIDTH)))   # vide `width`
		frame_height 		  = int(round(fvs.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # video `height
		total_frames      = int(round(fvs.stream.get(cv2.CAP_PROP_FRAME_COUNT)))   # video frames

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

			image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
			pose_landmarker_result = landmarker.detect_for_video(image, count)
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

			# ----------------- VISUALIZE 2D POSE -----------------
			if use_rerun is True:
				landmark_array = np.array ([ (	frame_width  * landmarks[lm].x, \
								frame_height * landmarks[lm].y, \
							-	(landmarks[lm].z) ) \
								for lm in range(len(landmarks))])

				landmark_array_3d = np.array ([( landmarks[lm].x, \
									landmarks[lm].y, \
									landmarks[lm].z) \
								for lm in range(len(landmarks))])  	# noqa: E101

				index_as_list =  np.array( list( kpts.items() ) )
				rr.log("person", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)

				# --------------------------
				# rr.set_time_seconds("time", time.time())
				rr.set_time_sequence("frame_idx", count)
				rr.log("video/rgb", rr.Image(frame).compress(jpeg_quality=80))

				# Log 2D Keypoints, Image and Mask
				rr.log("video/pose/points", rr.Points2D(landmark_array , class_ids=0, keypoint_ids=index_as_list[:,1]))

				rr.log("person/pose/points", rr.Points3D(landmark_array_3d , keypoint_ids=index_as_list[:,1]))

				rr.log("segmentation/frame", rr.Image(pose_landmarker_result.segmentation_masks[0].numpy_view()))
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
				pose[i] = ( 	round( (np.multiply(pose[i][0], frame_width)), 2),
						round( (np.multiply(pose[i][1], frame_height)), 2) )

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

			elif generateFormat == 'Original_COCO17':
				"""Generate JSON compatible with Original_COCO17 format
				"""
				onlyList              = []
						# INDEX   0, 1, 2, 3, 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16
				semGCN_reorder_fromMP = [00, 1, 4, 3, 6, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

				onlyList   = [pose[i] for i in semGCN_reorder_fromMP]
				score_list = [np.round(score[i], 3) for i in semGCN_reorder_fromMP]

				# update the score value
				score  = [[i] for i in score_list]

			elif generateFormat == 'COCO17':
				"""Generate JSON compatible with OpenPose (Body25) format
				"""
				onlyList            = []
						# INDEX   0, 1, 2, 3, 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16
				coco17_order_fromMP = [00, 12, 14, 16, 11, 13, 15, 24, 26, 27, 23, 25, 28, 5, 2, 8, 7]

				onlyList   = [pose[i] for i in coco17_order_fromMP ]
				score_list = [np.round(score[i], 2) for i in coco17_order_fromMP ]

				# replace the temp values with actual keypoints (that are not calculated in Mediapipe)
				onlyList[0] = HIP_MID
				onlyList[1] = NECK
				onlyList[7] = SPINE

				# update the score value
				score    = [[i] for i in score_list]

			# ----------------- Caclculating bbox -----------------

			# finding maximum and minimum of each onlyList columns
			tempList = np.array(onlyList)
			x_max = np.max(tempList[:,0])
			x_min = np.min(tempList[:,0])
			y_max = np.max(tempList[:,1])
			y_min = np.min(tempList[:,1])

			# calculating width and height of the bounding box
			bbox_width  = x_max - x_min
			bbox_height = y_max - y_min

			# calculating bounding box from min and max values
			bbox = [x_min, y_min, x_min+bbox_width, y_min+bbox_height]

			# ----------------- GENERATE JSON -----------------
			qq = {
				"frame_index":count+1,
				"skeleton":[
						{
							"pose" :onlyList,
							"score":score,
							"bbox" :bbox
						}
					]
				}
			json_data["data"].append(qq)

		json_filename = video_label + ".json"
		json_filename = json_filename.replace(".png","_keypoints")
		with open(json_filename, 'w') as fl:
			print("[INFO] Writing JSON file: ", json_filename)
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

	I   = np.array([1,2,3,4,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1 # start points  # noqa: E741
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
