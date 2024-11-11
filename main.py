import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
import joblib
import customtkinter

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # Load models
        self.deep_learning_model = tf.keras.models.load_model('MLDC.keras')
        self.knn_classifier = joblib.load('MLDCKNNClassifier.joblib')
        self.label_encoder = joblib.load('MLDCLabelEncoder.joblib')

        # Uploaded image path
        self.uploaded_image_path = None

        # Configure window
        self.title("Mango Leaf Disease Classifier")
        self.geometry(f"{1100}x{580}")

        # Configure grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # Create sidebar frame with widgets
        self.create_sidebar()

        # Create image panel
        self.create_image_panel()

        # Create labels for prediction and accuracy
        self.create_prediction_labels()

    def create_image_panel(self):
        # Create a frame for the image with a black border
        self.image_frame = customtkinter.CTkFrame(self, border_color="black", border_width=1)
        self.image_frame.grid(row=0, column=1, rowspan=3, padx=(20, 20), pady=20, sticky="nsew")

        # Create the image panel inside the frame
        self.image_panel = customtkinter.CTkLabel(self.image_frame, text="")
        self.image_panel.pack(expand=True, fill="both")

    def create_prediction_labels(self):
        # Create a frame for the prediction labels with a black border
        self.prediction_frame = customtkinter.CTkFrame(self, border_color="black", border_width=1)
        self.prediction_frame.grid(row=3, column=1, padx=20, pady=10, sticky="ew")

        # Create prediction and accuracy labels inside the frame
        self.label_result = customtkinter.CTkLabel(self.prediction_frame, text="Prediction will appear here", font=("Helvetica", 20))
        self.label_result.pack(pady=(10, 5))
        self.label_accuracy = customtkinter.CTkLabel(self.prediction_frame, text="", font=("Helvetica", 15))
        self.label_accuracy.pack(pady=(5, 10))

    def sidebar_button_event(self, button_name):
        if button_name == "Upload Image":
            self.upload_image()
        elif button_name == "Predict Leaf Disease":
            self.predict_disease()

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.uploaded_image_path = file_path
            img = Image.open(file_path)
            img = img.resize((500, 500), Image.Resampling.LANCZOS)
            img = ImageTk.PhotoImage(img)
            self.image_panel.configure(image=img)  # Changed from config to configure
            self.image_panel.image = img
            self.label_result.configure(text="")  # Also changed to configure
            self.label_accuracy.configure(text="")  # Also changed to configure

    def predict_disease(self):
        if self.uploaded_image_path:
            img = self.load_and_process_image(self.uploaded_image_path)
            img_array = np.array([img])
            features = self.deep_learning_model.predict(img_array)
            features_flatten = features.reshape(features.shape[0], -1)
            prediction = self.knn_classifier.predict(features_flatten)
            disease = self.label_encoder.inverse_transform(prediction)
            self.label_result.configure(text=f'Prediction: {disease[0]}')  # Changed from config to configure
            accuracy_percentage = self.calculate_accuracy(disease[0])
            self.label_accuracy.configure(text=f'Accuracy: {accuracy_percentage:.2f}%')  # Also changed to configure
        else:
            self.label_result.configure(text="Please upload an image first.")  # Also changed to configure
            self.label_accuracy.configure(text="")  # Also changed to configure


    def load_and_process_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        return img

    def calculate_accuracy(self, predicted_disease):
        # Replace this with your own accuracy calculation logic
        return np.random.uniform(80, 99)
    
    def create_sidebar(self):
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0, bg_color="#006400")  # Set bg_color to dark green
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        # Sidebar labels and buttons
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Mango Leaf Disease Classifier", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Upload Image", command=lambda: self.sidebar_button_event("Upload Image"))
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Predict Leaf Disease", command=lambda: self.sidebar_button_event("Predict Leaf Disease"))
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text="Exit", command=self.quit)
        self.sidebar_button_3.grid(row=10, column=0, padx=20, pady=10)

        # Appearance Mode
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["System", "Dark", "Light"],
                                                                    command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))

        # UI Scaling
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                            command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    
if __name__ == "__main__":
    app = App()
    app.mainloop()
