Traffic Sign Recognition Project
Overview
This project utilizes a neural network to classify traffic signs. The model is trained on a traffic sign dataset and is capable of predicting the type of traffic sign from a given image. A simple Tkinter-based GUI allows users to upload an image and view the prediction.

Folder Structure
traffic.py: Python script for training the model.
best_model.h5: The saved version of the best-trained model.
predict_sign.py: Python script with a Tkinter GUI for uploading an image and displaying the prediction.
images/: Directory containing 10 sample traffic sign images (used for testing).

Installation and Setup
Install dependencies:

Install TensorFlow:
pip install tensorflow
Install Tkinter (if not already installed):
sudo apt-get install python3-tk


Training the Model
The traffic.py file contains the code for training the neural network on the traffic sign dataset.
Run the following command to train the model and save the trained model as best_model.h5:
python traffic.py data/gtsrb/gtsrb
python predict_sign.py