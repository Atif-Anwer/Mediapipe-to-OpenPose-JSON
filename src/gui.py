# """
# GUI using CustomTkinter for generating OpenPose JSON using Mediapipe Pose model.
# """
import random
import tkinter
import tkinter.messagebox

# Importing the custom tkinter library for constructing the advanced GUI interface
import customtkinter
from PIL import Image, ImageTk

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

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
        self.grid_rowconfigure(0, weight=0)
        # self.grid_columnconfigure(1, weight=1)
        # self.grid_columnconfigure((0, 1), weight=1)

        # --- Create Frames ---
        self.sidebar_frame = customtkinter.CTkFrame(self, width=350, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.sidebar_frame.grid_rowconfigure(0, weight=0)

        self.kpt_frame = customtkinter.CTkFrame(self, height=600, width=600, corner_radius=0)
        self.kpt_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.sidebar_frame.grid_rowconfigure(0, weight=1)

        self.preview_frame = customtkinter.CTkFrame(self, height=400, width=400, corner_radius=0)
        self.preview_frame.grid(row=1, column=0, rowspan=4, sticky="nsew", padx=10, pady=10)
        self.sidebar_frame.grid_rowconfigure(0, weight=0)

        # --- Create Button Widgets ---
        self.btn_loadImage = customtkinter.CTkButton(self.sidebar_frame, command=self.ftn_loadImage, text="Load Image", width=275, height= 50)
        self.btn_loadImage.grid(row=1, column=0, padx=20, pady=10,)

        self.btn_loadJSON = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_event, text="Save JSON", width=275, height= 50)
        self.btn_loadJSON.grid(row=2, column=0, padx=20, pady=10)

        self.btn_plotOpenposeJSON = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_event, text="Load Openpose JSON", width=275, height= 50)
        self.btn_plotOpenposeJSON.grid(row=3, column=0, padx=20, pady=10)

        # Creating the frame
        # self.imgframe = customtkinter.CTkFrame(master = self.kpt_frame)
        # self.imgframe.pack(pady = 20, padx = 60, fill = "both", expand = True)

        # button.pack()

    def open_input_dialog_event(self):
        dialog = customtkinter.CTkInputDialog(text="Type in a number:", title="CTkInputDialog")
        print("CTkInputDialog:", dialog.get_input())

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def sidebar_button_event(self):
        print("sidebar_button click")

    # Creating the function for displaying our image
    def ftn_loadImage( self ):
        IMAGE_PATH = 'images/A-pose.png'
        img = Image.open(IMAGE_PATH)
        img = ImageTk.PhotoImage(img.resize((200, 200)))
        label = customtkinter.CTkLabel(master = self.preview_frame, image = img, text="")
        label.pack()

        IMAGE_PATH = 'images/A-pose.png'
        img2 = Image.open(IMAGE_PATH)
        img2 = ImageTk.PhotoImage(img2.resize((700, 700)))
        label2 = customtkinter.CTkLabel(master = self.kpt_frame, image = img2, text="")
        label2.pack()

    def clear_frame( self ):
        for widgets in self.frame.winfo_children():
            widgets.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
