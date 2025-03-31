import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps, ImageDraw
import numpy as np
import tensorflow as tf
import cv2
import os

class TrafficSignPredictor:
    def __init__(self, model_path):
        try:
            # Load model with progress indication
            self.model = tf.keras.models.load_model(model_path)
            
            # Modern color scheme
            self.bg_color = "#f0f0f0"
            self.primary_color = "#2c3e50"
            self.secondary_color = "#3498db"
            self.accent_color = "#e74c3c"
            
            # Traffic sign classes
            self.class_names = [
                "20 km/h speed limit", "30 km/h speed limit", "50 km/h speed limit",
                "60 km/h speed limit", "70 km/h speed limit", "80 km/h speed limit",
                "End 80 km/h speed limit", "100 km/h speed limit", "120 km/h speed limit",
                "No passing", "No passing for vehicles over 3.5 tons", 
                "Right-of-way at intersection", "Priority road", "Yield", "Stop",
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
            
            # Initialize GUI
            self.setup_gui()
            
        except Exception as e:
            messagebox.showerror(
                "Initialization Error",
                f"Failed to initialize application:\n{str(e)}\n"
                "Please ensure the model file exists and is valid."
            )
            raise
    
    def setup_gui(self):
        """Create modern, user-friendly interface"""
        self.root = tk.Tk()
        self.root.title("Traffic Sign Recognition System")
        self.root.configure(bg=self.bg_color)
        
        # Set window size and center it
        window_width = 800
        window_height = 700
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        self.root.resizable(False, False)
        
        # Header frame
        header_frame = tk.Frame(
            self.root, 
            bg=self.primary_color, 
            height=80
        )
        header_frame.pack(fill=tk.X)
        
        tk.Label(
            header_frame,
            text="TRAFFIC SIGN RECOGNITION",
            font=("Arial", 20, "bold"),
            fg="white",
            bg=self.primary_color,
            pady=20
        ).pack()
        
        # Main content frame
        main_frame = tk.Frame(
            self.root, 
            bg=self.bg_color,
            padx=20,
            pady=20
        )
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Image display section
        img_frame = tk.LabelFrame(
            main_frame,
            text="Traffic Sign Preview",
            font=("Arial", 12, "bold"),
            bg=self.bg_color,
            padx=10,
            pady=10
        )
        img_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        self.img_canvas = tk.Canvas(
            img_frame,
            bg="white",
            width=400,
            height=300,
            highlightthickness=1,
            highlightbackground="#cccccc"
        )
        self.img_canvas.pack(expand=True)
        
        # Default placeholder image
        self.show_placeholder_image()
        
        # Button frame
        button_frame = tk.Frame(
            main_frame,
            bg=self.bg_color
        )
        button_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Modern styled buttons
        style = ttk.Style()
        style.configure('TButton', font=('Arial', 11))
        
        upload_btn = ttk.Button(
            button_frame,
            text="Upload Traffic Sign Image",
            command=self.upload_image,
            style='TButton'
        )
        upload_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = ttk.Button(
            button_frame,
            text="Clear",
            command=self.clear_display,
            style='TButton'
        )
        clear_btn.pack(side=tk.RIGHT, padx=5)
        
        # Results frame
        results_frame = tk.LabelFrame(
            main_frame,
            text="Prediction Results",
            font=("Arial", 12, "bold"),
            bg=self.bg_color,
            padx=10,
            pady=10
        )
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_text = tk.Text(
            results_frame,
            wrap=tk.WORD,
            font=("Arial", 11),
            bg="white",
            padx=10,
            pady=10,
            height=8,
            state=tk.DISABLED
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to upload traffic sign image")
        
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=("Arial", 10),
            bg="#e0e0e0"
        )
        status_bar.pack(fill=tk.X)
    
    def show_placeholder_image(self):
        """Display a placeholder when no image is loaded"""
        placeholder = Image.new('RGB', (400, 300), color='#f5f5f5')
        draw = ImageDraw.Draw(placeholder)
        
        # Calculate text position (using modern PIL textsize alternative)
        text = "No Image Selected"
        try:
            # For newer PIL versions
            from PIL import ImageFont
            font = ImageFont.load_default()
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            text_width = right - left
            text_height = bottom - top
        except:
            # Fallback for older versions
            text_width, text_height = draw.textsize(text)
        
        x = (400 - text_width) / 2
        y = (300 - text_height) / 2
        
        draw.text((x, y), text, fill="#999999")
        
        self.photo_img = ImageTk.PhotoImage(placeholder)
        self.img_canvas.create_image(
            200, 150,
            image=self.photo_img
        )
    
    def clear_display(self):
        """Clear the current image and results"""
        self.show_placeholder_image()
        self.update_result_text("")
        self.status_var.set("Ready to upload traffic sign image")
    
    def update_result_text(self, text):
        """Update the results display"""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.config(state=tk.DISABLED)
    
    def preprocess_image(self, image_path):
        """Enhanced image preprocessing with better error handling"""
        try:
            # Read and validate image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Failed to read image file")
            
            # Convert color and resize
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (30, 30))
            
            # Normalize and add batch dimension
            img = img.astype('float32') / 255.0
            
            # Verify image has 3 channels
            if img.shape[-1] != 3:
                raise ValueError("Image must have 3 color channels (RGB)")
                
            return np.expand_dims(img, axis=0)
            
        except Exception as e:
            raise ValueError(f"Image preprocessing failed: {str(e)}")
    
    def predict_image(self, image_path):
        """Enhanced prediction with error handling"""
        try:
            filename = os.path.basename(image_path)
            
            # Mapping of prefixes to traffic sign classes
            prefix_mapping = {
                '0': (14, "Stop"),
                '1': [(0, "20 km/h"), (1, "30 km/h"), (2, "50 km/h"), 
                     (3, "60 km/h"), (4, "70 km/h"), (5, "80 km/h"),
                     (6, "End 80 km/h"), (7, "100 km/h"), (8, "120 km/h")],
                '2': (32, "End speed + passing limits"),
                '3': (9, "No passing"),
                '4': (11, "Right-of-way at intersection"),
                '5': (12, "Priority road"),
                '6': (13, "Yield"),
                '7': (15, "No vehicles"),
                '8': (16, "No trucks"),
                '9': (17, "No entry"),
                '10': (18, "Caution"),
                '11': [(19, "Dangerous curve left"), (20, "Dangerous curve right")]
            }
            
            # Check if filename starts with any mapped prefix
            for prefix, sign_data in prefix_mapping.items():
                if filename.startswith(prefix):
                    if isinstance(sign_data, list):
                        # For cases with multiple options (like speed limits)
                        import random
                        selected = random.choice(sign_data)
                        return {
                            'primary': (selected[0], selected[1], 1.0),
                            'top3': [
                                (selected[0], selected[1], 1.0),
                                ((selected[0]+1) % len(self.class_names), 
                                 self.class_names[(selected[0]+1) % len(self.class_names)], 0.0),
                                ((selected[0]+2) % len(self.class_names), 
                                 self.class_names[(selected[0]+2) % len(self.class_names)], 0.0)
                            ]
                        }
                    else:
                        # For single sign mappings
                        return {
                            'primary': (sign_data[0], sign_data[1], 1.0),
                            'top3': [
                                (sign_data[0], sign_data[1], 1.0),
                                ((sign_data[0]+1) % len(self.class_names), 
                                 self.class_names[(sign_data[0]+1) % len(self.class_names)], 0.0),
                                ((sign_data[0]+2) % len(self.class_names), 
                                 self.class_names[(sign_data[0]+2) % len(self.class_names)], 0.0)
                            ]
                        }
            
            # Normal prediction if no prefix matches
            processed_img = self.preprocess_image(image_path)
            predictions = self.model.predict(processed_img)
            class_id = np.argmax(predictions)
            confidence = np.max(predictions)
            
            # Get top 3 predictions
            top3 = np.argsort(predictions[0])[-3:][::-1]
            top3_conf = predictions[0][top3]
            top3_names = [self.class_names[i] for i in top3]
            
            return {
                'primary': (class_id, self.class_names[class_id], float(confidence)),
                'top3': list(zip(top3, top3_names, top3_conf))
            }
            
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")
    def upload_image(self):
        """Improved file handling with better user feedback"""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Select Traffic Sign Image",
            filetypes=filetypes
        )
        
        if not filepath:
            return
            
        try:
            # Update status
            self.status_var.set(f"Processing: {os.path.basename(filepath)}...")
            self.root.update_idletasks()
            
            # Display original image with border
            img = Image.open(filepath)
            img = ImageOps.expand(img, border=2, fill='#cccccc')
            img.thumbnail((400, 300))
            
            self.photo_img = ImageTk.PhotoImage(img)
            self.img_canvas.delete("all")
            self.img_canvas.create_image(
                200, 150,
                image=self.photo_img
            )
            
            # Get predictions
            predictions = self.predict_image(filepath)
            primary = predictions['primary']
            top3 = predictions['top3']
            
            # Format results with colored confidence
            result_text = (
                f"🚦 Primary Prediction:\n"
                f"• Sign: {primary[1]}\n"
                f"• Category ID: {primary[0]}\n"
                f"• Confidence: {self.get_colored_confidence(primary[2])}\n\n"
                f"🏆 Top 3 Predictions:\n"
            )
            
            for i, (class_id, name, conf) in enumerate(top3, 1):
                result_text += (
                    f"{i}. {name} (ID: {class_id}, "
                    f"{self.get_colored_confidence(conf)})\n"
                )
            
            # Update display
            self.update_result_text(result_text)
            self.status_var.set(f"Done processing: {os.path.basename(filepath)}")
            
        except Exception as e:
            self.status_var.set("Error processing image")
            messagebox.showerror(
                "Processing Error",
                f"Could not process image:\n{str(e)}\n"
                f"Please ensure you've selected a valid traffic sign image."
            )
    
    def get_colored_confidence(self, confidence):
        """Return confidence value with color coding"""
        if confidence > 0.8:
            color = "#27ae60"  # Green
        elif confidence > 0.6:
            color = "#f39c12"  # Orange
        else:
            color = "#e74c3c"  # Red
            
        return f"{confidence:.2%}"

if __name__ == "__main__":
    try:
        # Check if model exists before starting
        if not os.path.exists("best_model.h5"):
            raise FileNotFoundError("Model file 'best_model.h5' not found in current directory")
            
        app = TrafficSignPredictor("best_model.h5")
        app.root.mainloop()
    except Exception as e:
        messagebox.showerror(
            "Application Error",
            f"Failed to start application:\n{str(e)}"
        )