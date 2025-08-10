import os
import numpy as np
from PIL import Image


def sigmoid(z):
    # z is a scalar or numpy array of any size.

    s = 1 / (1 + np.exp(-z))
    return s


def forward_propagation(X, parameters):
    # X is the input data of size (n_x, m)
    # parameters is a python dictionary containing the parameters "W1", "b1", "W2", "b2"

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    
    return A2


def predict(parameters, X):
    # Using the learned parameters W1, b1, W2, b2 to predict the labels for a given dataset X
    # parameters is a dictionary containing the learned parameters 
    # X is the input data of size (n_x, m)

    A2 = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    
    return predictions


def image_test(my_image):
    num_px = 64
    cache = np.load("model_params.npz")
    params = {"W1": cache["W1"], "b1": cache["b1"], "W2": cache["W2"], "b2": cache["b2"]}

    fname = "images/" + my_image
    image = np.array(Image.open(fname).resize((num_px, num_px)))
    image = image / 255.
    image = image.reshape((1, num_px * num_px * 3)).T

    predicted_image = predict(params, image)
    if np.squeeze(predicted_image):
        return "It's a cat"
    else:
        return "It's not a cat"


import os

def select_file(folder_path):
    # Shows only .jpg, .jpeg, and .png files in the given folder path
    # and lets the user select one

    try:
        # Get list of image files only
        valid_extensions = {".jpg", ".jpeg", ".png"}
        files = [
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f)) and
               os.path.splitext(f.lower())[1] in valid_extensions
        ]

        if not files:
            print(f"No .jpg, .jpeg, or .png files found in: {folder_path}")
            return None

        # Display the files
        print("Image files in the folder:\n")
        for i, file in enumerate(files, start=1):
            print(f"{i}. {file}")

        # User selects the file
        while True:
            choice = input("\nSelect the file number: ")
            if choice == "0":
                return None
            if 1 <= int(choice) <= len(files):
                selected_file = files[int(choice) - 1]
                print(f"\nYou selected: {selected_file}")
                return selected_file
            else:
                print("Invalid choice! Please enter a number between 1 and", len(files))

    except FileNotFoundError:
        print(f"Folder does not exist: {folder_path}")
        return None


# initial menu
while True:
    print("Enter 0 to exit")
    file = select_file("images/")
    if file is None:
        print("Exiting...")
        break
    else:
        print(image_test(file))