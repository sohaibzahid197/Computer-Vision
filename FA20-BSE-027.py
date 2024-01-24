import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Function to load and preprocess images from a given folder
def load_preprocess_images(directory, size=(128, 128)):
    all_images = []
    all_labels = []

    # Mapping of class names to numeric labels
    label_mapping = {"500": 0, "1000": 1, "5000": 2}

    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)

        if not os.path.isdir(folder_path):
            continue

        label = label_mapping.get(folder)
        if label is None:
            continue

        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)

            if not os.path.isfile(img_path):
                continue

            img = cv2.imread(img_path)
            img = cv2.resize(img, size)
            img = img / 255.0

            all_images.append(img)
            all_labels.append(label)

    return np.array(all_images), np.array(all_labels)

# Load and preprocess the dataset
data_folder = "E:\CV-2\augmented-dataset\augmented-dataset"
processed_images, processed_labels = load_preprocess_images(data_folder)

# Split the dataset into training and testing sets
train_imgs, test_imgs, train_lbls, test_lbls = train_test_split(
    processed_images, processed_labels, test_size=0.2, random_state=42
)

# Function to create a CNN model
def create_cnn(activation):
    cnn_model = models.Sequential()

    # Configure layers based on the activation function
    if activation == "relu":
        cnn_model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)))
        cnn_model.add(layers.MaxPooling2D((2, 2)))
        cnn_model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        cnn_model.add(layers.MaxPooling2D((2, 2)))
        cnn_model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    else:
        cnn_model.add(layers.Conv2D(32, (3, 3), activation="sigmoid", input_shape=(128, 128, 3)))
        cnn_model.add(layers.MaxPooling2D((2, 2)))
        cnn_model.add(layers.Conv2D(64, (3, 3), activation="sigmoid"))
        cnn_model.add(layers.MaxPooling2D((2, 2)))
        cnn_model.add(layers.Conv2D(64, (3, 3), activation="sigmoid"))

    cnn_model.add(layers.Flatten())
    cnn_model.add(layers.Dense(64, activation="sigmoid"))
    cnn_model.add(layers.Dense(3, activation="softmax"))

    cnn_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return cnn_model

# Function to train and evaluate the model
def train_evaluate_model(activation_choice):
    neural_net = create_cnn(activation_choice)

    neural_net.fit(train_imgs, train_lbls, epochs=10, batch_size=32, validation_data=(test_imgs, test_lbls))

    _, model_accuracy = neural_net.evaluate(test_imgs, test_lbls, verbose=0)
    return neural_net

# Setup the main application window
app_window = tk.Tk()
app_window.title("Currency Recognition System")
app_window.geometry("400x300")  # Setting the window size

# Styling
style = ttk.Style()
style.theme_use('clam')
style.configure('TFrame', background='#333333')
style.configure('TLabel', background='#333333', foreground='white')
style.configure('TRadiobutton', background='#333333', foreground='white')
style.configure('TButton', background='#4CAF50', foreground='white', font=('Helvetica', 10, 'bold'))

# Function to classify a selected image
def classify_currency():
    img_file = filedialog.askopenfilename()
    if not img_file:
        return

    image_to_classify = preprocess_single_image(img_file)

    chosen_activation = activation_choice.get()
    trained_model = train_evaluate_model(chosen_activation)

    prediction_probs = trained_model.predict(image_to_classify)
    predicted_class = np.argmax(prediction_probs)

    currency_labels = ["500 PKR", "1000 PKR", "5000 PKR"]
    result.config(text=f"Predicted Currency: {currency_labels[predicted_class]}")

# Function to load and preprocess a single image
def preprocess_single_image(path, size=(128, 128)):
    single_image = cv2.imread(path)
    single_image = cv2.resize(single_image, size)
    single_image = single_image / 255.0
    return np.expand_dims(single_image, axis=0)

# Create and layout widgets
main_frame = ttk.Frame(app_window)
main_frame.pack(expand=True, fill='both')

app_title = ttk.Label(main_frame, text="Currency Recognition System", font=("Helvetica", 18, 'bold'))
app_title.pack(pady=10)

activation_prompt = ttk.Label(main_frame, text="Choose Activation Function:")
activation_prompt.pack(pady=5)

activation_choice = tk.StringVar()
activation_choice.set("sigmoid")

sigmoid_option = ttk.Radiobutton(main_frame, text="Sigmoid", variable=activation_choice, value="sigmoid")
relu_option = ttk.Radiobutton(main_frame, text="ReLU", variable=activation_choice, value="relu")

sigmoid_option.pack()
relu_option.pack()

classify_btn = ttk.Button(main_frame, text="Classify Image", command=classify_currency)
classify_btn.pack(pady=10)

result = ttk.Label(main_frame, text="")
result.pack(pady=10)

# Configure window layout
app_window.columnconfigure(0, weight=1)
app_window.rowconfigure(0, weight=1)

# Run the application
app_window.mainloop()