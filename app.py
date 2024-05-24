from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from PIL import Image
import pickle
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras.applications import InceptionV3
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import random
import os
from keras import regularizers

def model_load():
    pwd = os.getcwd()

    dataset_path = pwd + "\\dataset.csv"

    df = pd.read_csv(dataset_path)

    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)

    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=20,
    )

    batch_size = 32
    img_height = 224
    img_width = 224

    # Get a batch of test data
    test_gen = datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='image',
        y_col='gender_index',  # Assuming integer labels for gender
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='raw',  # Handle integer labels
        shuffle=False  # Don't shuffle during testing for correct image order
    )

    model_CNN = keras.models.load_model(r"C:\Users\sweek\OneDrive - Lambton College\Projects Main\Age and Gender Prediction\CNN_model.keras")

    test_loss, test_accuracy = model_CNN.evaluate(test_gen)

    return model_CNN

model_CNN = None

if model_CNN == None:
    model_CNN=model_load()

    app = Flask(__name__, template_folder='.')

    batch_size = 32
    img_height = 224
    img_width = 224

    def preprocess_image(image):
        # Preprocess the image here, according to your model's requirements
        image = image.resize((img_height, img_width))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image /= 255.0  # Rescale pixel values to [0, 1]
        return image

    # Define route for the home page
    @app.route('/')
    def home():
        return render_template('app.html')

    # Define route to handle image upload and make predictions
    @app.route('/predict', methods=['POST'])
    def predict():
        # Get the uploaded image file
        file = request.files['file']
        # Read the image file
        image = Image.open(file)
        # Preprocess the image
        image_array = preprocess_image(image)
        # Make predictions
        prediction = model_CNN.predict(image_array)
        # Convert the prediction to text
        gender = 'Female' if prediction > 0.5 else 'Male'
        # Return the predicted gender as JSON
        return jsonify({'gender': gender})

    if __name__ == '__main__':
        app.run(debug=True)

else:
    app = Flask(__name__, template_folder='.')

    batch_size = 32
    img_height = 224
    img_width = 224

    def preprocess_image(image):
        # Preprocess the image here, according to your model's requirements
        image = image.resize((img_height, img_width))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image /= 255.0  # Rescale pixel values to [0, 1]
        return image

    # Define route for the home page
    @app.route('/')
    def home():
        return render_template('app.html')

    # Define route to handle image upload and make predictions
    @app.route('/predict', methods=['POST'])
    def predict():
        # Get the uploaded image file
        file = request.files['file']
        # Read the image file
        image = Image.open(file)
        # Preprocess the image
        image_array = preprocess_image(image)
        # Make predictions
        prediction = model_CNN.predict(image_array)
        # Convert the prediction to text
        gender = 'Female' if prediction > 0.5 else 'Male'
        # Return the predicted gender as JSON
        return jsonify({'gender': gender})

    if __name__ == '__main__':
        app.run(debug=True)
