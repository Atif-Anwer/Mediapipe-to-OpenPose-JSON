<a name="readme"></a>

<!-- [![Contributors][contributors-shield]][contributors-url] -->
![Python][python-shield]

<!-- GETTING STARTED -->
## Info

Mediapipe pose extraction and exporting to OpenPose format but Mediapipe has 33 keypoints as output as compared to 25 from Openpose. The keypoints also have a different order.
The code below extracts the keypoints from images in a folder and exports them as an Openpose JSON format with 25 keypoints.
the JSON is compaitble with SMPLify-X for 3D shape extraction
- Uses Hydra.cc for config
- Supports only 1 person per frame (Mediapipe limitation)
- Supports multiple image extensions in folder (PNG, JPG, JPEG etc)

### INPUT:
- Folder Path Containing RGB Images

### RETURNS:
- Mediapipe outputkeypoints in OpenPose compatible JSON format

## Requirements
- Hydra.cc
- Mediapipe


<!-- MARKDOWN LINKS & IMAGES -->
[python-shield]: https://img.shields.io/badge/Python-3.7-blue?style=for-the-badge&logo=appveyor
[tf-shield]: https://img.shields.io/badge/Tensorflow-2.8-orange?style=for-the-badge&logo=appveyor

[issues-shield]: https://img.shields.io/github/issues/Atif-Anwer/SpecSeg?style=for-the-badge
[issues-url]: https://github.com/Atif-Anwer/SpecSeg/issues
[license-shield]: https://img.shields.io/badge/License-CC-brightgreen?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/atifanwer/

<!-- Soruce: https://github.com/othneildrew/Best-README-Template/pull/73 -->