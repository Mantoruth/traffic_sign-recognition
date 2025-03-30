import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        shear_range=0.1,
        brightness_range=[0.8,1.2],
        fill_mode='nearest',
        horizontal_flip=False
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]

    # Fit model
    model.fit(
        datagen.flow(x_train, y_train),
        validation_data=(x_test, y_test),
        epochs=30,
        callbacks=callbacks
    )

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)  # <- This line was misaligned

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
    else:
        filename = "best_model.h5"
    
    model.save(filename)
    print(f"Model saved to {filename}")

def load_data(data_dir):
    images = []
    labels = []
    print(f"\nLoading data from: {os.path.abspath(data_dir)}")
    
    for category in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(category))
        if not os.path.isdir(category_dir):
            print(f"⚠️ Missing category folder: {category}")
            continue
            
        image_files = [f for f in os.listdir(category_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm'))]
        
        print(f"Category {category}: Found {len(image_files)} images")
        
        for image_file in image_files:
            image_path = os.path.join(category_dir, image_file)
            try:
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError("Failed to read image")
                    
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype('float32') / 255.0
                
                images.append(img)
                labels.append(category)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
    
    print(f"\nTotal images loaded: {len(images)}")
    if len(images) == 0:
        raise ValueError("No images were loaded - check your data directory path")
    
    return np.array(images), np.array(labels)

def get_model():
    model = tf.keras.models.Sequential([
        # First Conv Block
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', 
                              input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.2),
        
        # Second Conv Block
        tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.3),
        
        # Third Conv Block
        tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.4),
        
        # Dense Layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    main()