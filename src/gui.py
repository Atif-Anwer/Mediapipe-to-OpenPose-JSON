# """
# GUI using CustomTkinter for generating OpenPose JSON using Mediapipe Pose model.
# """
import json
import logging
import os
import pathlib
import random
import tkinter
import tkinter.messagebox
from glob import glob
from os.path import join
from pickle import FALSE
from tkinter import filedialog as fd
from tkinter import messagebox

# Importing the custom tkinter library for constructing the advanced GUI interface
import customtkinter
from mediapipe_JSON import generate_MP_JSON
from natsort import natsorted
from PIL import Image, ImageTk
from plot_json import plot_OpenposeJSON

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        # Configuring python logger
        self.log = logging.getLogger(__name__)

        # configure window
        self.title("Mediapipe to OpenPose JSON Generator")

        width=1050
        height=1000
        # screenwidth = (self.winfo_screenwidth())
        # screenheight = self.winfo_screenheight()
        # self.geometry(f"{(screenwidth)}x{(screenheight)}")
        self.geometry(f"{(width)}x{(height)}")

        # configure grid layout (4x4)
        # weight of 1 so it will expand. weight of 0 and will only be as big as it needs to be to fit the widgets inside of it.
        # self.grid_columnconfigure((2, 3), weight=0)
        # self.grid_rowconfigure((0, 1, 2), weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        # self.grid_columnconfigure((0, 1), weight=1)

        # --- Create Frames ---
        self.sidebar_frame = customtkinter.CTkFrame(self, width=350, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.sidebar_frame.grid_rowconfigure(0, weight=0)

        self.kpt_frame = customtkinter.CTkFrame(self, height=600, width=600, corner_radius=0)
        self.kpt_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10, rowspan=2)
        self.sidebar_frame.grid_rowconfigure(0, weight=1)

        self.preview_frame = customtkinter.CTkFrame(self, height=400, width=400, corner_radius=0)
        self.preview_frame.grid(row=1, column=0, rowspan=4, sticky="nsew", padx=10, pady=10)
        self.sidebar_frame.grid_rowconfigure(0, weight=0)

        # --- Create Button Widgets ---
        self.btn_loadImage = customtkinter.CTkButton(self.sidebar_frame, command=self.ftn_loadImage, text="Load Image", width=275, height= 50)
        self.btn_loadImage.grid(row=1, column=0, padx=20, pady=10,)

        self.btn_plotOpenposeJSON = customtkinter.CTkButton(self.sidebar_frame, command=self.ftn_loadOpenPoseJSON, text="Plot Openpose JSON", width=275, height= 50)
        self.btn_plotOpenposeJSON.grid(row=3, column=0, padx=20, pady=10)


    def ftn_loadOpenPoseJSON(self) -> None:
        """
        Genreate Openpose JSON from Mediapipe JSON files in a folder
        """

        self.clearFrame(self.preview_frame)
        self.clearFrame(self.kpt_frame)

        dirpath = fd.askdirectory(title="Directory containing OpenPose JSON Files", initialdir='./images')
        img_folder = []
        # img_folder.extend(glob(join(str(dirpath), ext)))
        img_folder = list(pathlib.Path(dirpath).glob('*.json'))

        for path in natsorted(img_folder):
            # Debug print file name
            self.log.info(f"Processing image file: {path}")

            JSON_PATH = path
            OUTPUT_PATH = pathlib.Path(str(path).replace(".json","_OpenPoseKeypoints.jpg"))
            # filename=path.replace(".json","_OpenPoseKeypoints.json")
            # OUTPUT_PATH = dirpath + 'output.jpg'
            HT: int = 1500
            WD: int = 1500

            qq = os.getcwd()
            # os.chdir(qq+'/images/')
            os.system(f"python ./src/plot_json.py {JSON_PATH} {OUTPUT_PATH} {HT} {WD}")

            output_img = Image.open(OUTPUT_PATH)
            output_img = ImageTk.PhotoImage(output_img.resize((int(HT/2), int(WD/2))))
            label = customtkinter.CTkLabel(master = self.kpt_frame, image = output_img, text="")
            label.pack()

            answer = messagebox.askyesno("Next image?","Do you want to open the next JSON file?")

            if answer is False:
                break

            self.clearFrame(self.preview_frame)
            self.clearFrame(self.kpt_frame)




    # Creating the function for displaying our image
    def ftn_loadImage( self ) -> None:
        """
        Load image from a folder and generate Mediapipe JSON file
        """

        self.clearFrame(self.preview_frame)
        self.clearFrame(self.kpt_frame)

        dirpath = fd.askdirectory(title="Directory containing Image(s)", initialdir='./images')
        # IMAGE_PATH = 'images/A-pose.png'
        img_folder = []
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            img_folder.extend(glob(join(str(dirpath), ext)))

        qq = os.getcwd()
        # os.chdir(qq+'/images/')
        # Generate keypoints for all images in folder
        os.system(f"python src/mediapipe_JSON.py files.test_img_path={dirpath}")

        for path in natsorted(img_folder):
            # Debug print file name
            self.log.info(f"Processing image file: {path}")
            # IMAGE_PATH = filename
            img = Image.open(path)
            img = ImageTk.PhotoImage(img.resize((200, 200)))
            label = customtkinter.CTkLabel(master = self.preview_frame, image = img, text="")
            label.pack()

            filename=path.replace(".","_keypoints.")
            img2 = Image.open(filename)
            img2 = ImageTk.PhotoImage(img2.resize((700, 700)))
            label2 = customtkinter.CTkLabel(master = self.kpt_frame, image = img2, text="")
            label2.pack()

            answer = messagebox.askyesno("Next image?","Do you want to open the next Keypoint file?")

            if answer is False:
                break

            self.clearFrame(self.preview_frame)
            self.clearFrame(self.kpt_frame)



    def clearFrame( self, frame) -> None:
        """Clears the frame contents

        Args:
            frame (CTKFrame): Frame to be cleared
        """
        # destroy all widgets from frame
        for widget in frame.winfo_children():
            widget.destroy()

        # this will clear frame and frame will be empty
        # if you want to hide the empty panel then
        frame.pack_forget()


if __name__ == "__main__":
    app = App()
    app.mainloop()
