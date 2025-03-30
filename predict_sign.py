import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2
import os

class TrafficSignPredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = [
            "20 km/h speed limit", "30 km/h speed limit", "50 km/h speed limit",
            "60 km/h speed limit", "70 km/h speed limit", "80 km/h speed limit",
            "End 80 km/h speed limit", "100 km/h speed limit", "120 km/h speed limit",
            "No passing", "No passing for vehicles over 3.5 tons", 
            "Right-of-way at intersection", "Priority road", "Yield", "Stopp",
            "No vehicles", "No trucks", "No entry", "Caution", 
            "Dangerous curve left", "Dangerous curve right", 
            "Double curve", "Bumpy road", "Slippery road", 
            "Road narrows on right", "Road work", "Traffic signals",
            "Pedestrians", "Children crossing", "Bicycle crossing",
            "Beware of ice/snow", "Wild animals crossing",
            "End speed + passing limits", "Turn right ahead", 
            "Turn left ahead", "Ahead only", "Go straight or right",
            "Go straight or left", "Keep right", "Keep left", 
            "Roundabout mandatory", "End of no passing", 
            "End no passing for vehicles over 3.5 tons"
]
        # GUI setup
        self.root = tk.Tk()
        self.root.title("Traffic Sign Predictor")
        
        # Upload button
        tk.Button(
            self.root, 
            text="Upload Image", 
            command=self.upload_image,
            font=("Arial", 12),
            padx=10, pady=5
        ).pack(pady=20)
        
        # Image display
        self.img_label = tk.Label(self.root)
        self.img_label.pack()
        
        # Prediction result
        self.result_label = tk.Label(
            self.root, 
            text="", 
            font=("Arial", 14, "bold")
        )
        self.result_label.pack(pady=20)
        
    def preprocess_image(self, image_path):
        """Resize and normalize the image for the model"""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (30, 30))
        img = img / 255.0
        return np.expand_dims(img, axis=0)
    
    def predict_image(self, image_path):
        """Return (class_id, sign_name, confidence)"""
        processed_img = self.preprocess_image(image_path)
        predictions = self.model.predict(processed_img)
        class_id = np.argmax(predictions)
        confidence = np.max(predictions)
        return class_id, self.class_names[class_id], float(confidence)
    
    def upload_image(self):
        """Handle file upload and display prediction"""
        filepath = filedialog.askopenfilename(
            initialdir="images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if filepath:
            try:
                # Display image
                img = Image.open(filepath)
                img.thumbnail((300, 300))
                img_tk = ImageTk.PhotoImage(img)
                self.img_label.config(image=img_tk)
                self.img_label.image = img_tk
                
                # Get prediction
                class_id, sign, confidence = self.predict_image(filepath)
                
                # Show result
                self.result_label.config(
                    text=f"Prediction: {sign}\n"
                         f"Category: {class_id}\n"
                         f"Confidence: {confidence:.2%}",
                    fg="green" if confidence > 0.7 else "orange"
                )
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image:\n{str(e)}")

if __name__ == "__main__":
    predictor = TrafficSignPredictor("best_model.h5")
    predictor.root.mainloop()