<a name="readme"></a>

<!-- [![Contributors][contributors-shield]][contributors-url] -->
![Python][python-shield]

<!-- GETTING STARTED -->

# MediaPipe Pose to OpenPose JSON Generator
## Info

Mediapipe pose extraction and exporting to OpenPose format but Mediapipe has 33 keypoints as output as compared to 25 from Openpose. The keypoints also have a different order.
The code in this repository has three scripts:
- `mediapipe_JSON.py` :  extracts the keypoints from all images in a folder and exports them as an Openpose JSON format with 25 keypoints. The JSON is compaitble with SMPLify-X for 3D shape extraction.
- `plot_json.py` : plots the OpenPose keypoints and saves the image.
- `gui.py` : GUI (made in [CustomTkinter](https://customtkinter.tomschimansky.com)) for the above two scripts, that displays the Mediapipe keypoints on the loaded image as well as the generated OpenPose keypoints. **Note that each of the scripts can be run independently and the GUI is not required to be run.** But running the GUI simplifies the process of running the scripts.


![Rofi](./src//GUI.png "GUI Window" )

## Other features:
- Uses Hydra.cc for config
- Supports only 1 person per frame (Mediapipe limitation)
- Supports multiple image extensions in folder (PNG, JPG, JPEG etc)
- Each script can be run separately, gui is optional

## Requirements
To install all the required packages: `pip install -r requirements.txt`

## ToDo:
- [ ] Image overlay for OpenPose JSON plot
- [ ] Better GUI scaling for different screen sizes and images

<!-- MARKDOWN LINKS & IMAGES -->
[python-shield]: https://img.shields.io/badge/Python-3.7-blue?style=for-the-badge&logo=appveyor
