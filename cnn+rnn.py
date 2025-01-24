import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, SimpleRNN, Reshape # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.optimizers import SGD # type: ignore
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd

def load_az_dataset():
    dataset_path = r"C:\Users\USER\OneDrive\Documents\Desktop\DS+ML+NLP\Eminst data\A_Z Handwritten Data.csv"

    # Check if the file exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

    # Load the CSV file using pandas
    print("[INFO] Loading dataset...")
    df = pd.read_csv(dataset_path, header=None)

    # Split labels and image data
    labels = df.iloc[:, 0].values  # First column contains the labels
    data = df.iloc[:, 1:].values   # Remaining columns contain the pixel values

    # Reshape the image data into 28x28 arrays
    data = data.reshape(-1, 28, 28).astype("uint8")

    return np.array(data, dtype="float32"), np.array(labels, dtype="int")


# Load the MNIST dataset
def load_mnist_dataset():
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    data = np.vstack([train_data, test_data])
    labels = np.hstack([train_labels, test_labels])
    return data, labels

# Training process
def train_model():
    output_model = "handwriting_recognition_model_lstm_rnn_cnn.h5"
    plot_path = "training_plot_lstm_rnn_cnn.png"

    print("[INFO] Loading datasets...")
    az_data, az_labels = load_az_dataset()
    digits_data, digits_labels = load_mnist_dataset()

    az_labels += 10  # Offset labels for A-Z
    data = np.vstack([az_data, digits_data])
    labels = np.hstack([az_labels, digits_labels])

    data = [cv2.resize(img, (32, 32)) for img in data]
    data = np.array(data, dtype="float32") / 255.0
    data = np.expand_dims(data, axis=-1)

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    train_x, test_x, train_y, test_y = train_test_split(
        data, labels, test_size=0.2, stratify=labels, random_state=42
    )

    aug = ImageDataGenerator(
        rotation_range=10, zoom_range=0.05, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.15, horizontal_flip=False, fill_mode="nearest"
    )

    print("[INFO] Compiling model...")

    # Define the model
    model = Sequential()

    # CNN layers
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))

    # Flatten and reshape for LSTM/RNN input
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Reshape((4, 32)))  # Reshape to (timesteps, features), adjust as needed

    # LSTM and RNN layers
    model.add(LSTM(64, return_sequences=True, activation="tanh"))
    model.add(SimpleRNN(32, activation="tanh"))

    # Fully connected output layer
    model.add(Flatten())
    model.add(Dense(len(lb.classes_), activation="softmax"))

    # Compile the model
    opt = SGD(learning_rate=0.01, decay=0.01 / 2, momentum=0.9)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    print("[INFO] Training network...")
    history = model.fit(
        aug.flow(train_x, train_y, batch_size=128),
        validation_data=(test_x, test_y),
        steps_per_epoch=len(train_x) // 128,
        epochs=2,
        verbose=1
    )

    print("[INFO] Saving model...")
    output_model = "output_model.h5"  # Define the file name
    model.save(output_model)



    print("[INFO] Plotting training history...")
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 50), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 50), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 50), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, 50), history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plot_path)

if __name__ == "__main__":
    train_model()
