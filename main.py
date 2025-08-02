import os
import numpy as np
import copy
import h5py
from PIL import Image

def sigmoid(z):
    # z is a scalar or numpy array of any size.
    z = np.clip(z, -500, 500)
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    # dim is the size of the w vector or n_x

    w = np.zeros((dim, 1))
    b = 0.0
    return w, b


def propagate(w, b, X, Y):
    # w (weights) is a numpy array of size (num_px * num_px * 3, 1)
    # b (bias) is a scalar
    # X is data of size (num_px * num_px * 3, number of data)
    # Y is true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of data)

    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)
    epsilon = 1e-8 # epsilon to prevent log(0)
    cost = -np.sum(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon)) / m
    dw = (np.dot(X, (A - Y).T)) / m
    db = (np.sum(A - Y)) / m

    cost = np.squeeze(np.array(cost))
    grads = {"dw": dw,
             "db": db}
    return grads, cost


def optimize(w, b, X, Y, num_iterations=1000, learning_rate=0.5, print_cost=False):
    # w (weights) is a numpy array of size (num_px * num_px * 3, 1)
    # b (bias) is a scalar
    # X is data of shape (num_px * num_px * 3, number of data)
    # Y is the true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of data)

    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    dw, db = 0, 0
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w -= learning_rate * dw
        b -= learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

            # prints the cost every 100 training iterations
            if print_cost:
                print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    # w (weights) is a numpy array of size (num_px * num_px * 3, 1)
    # b (bias) is a scalar
    # X is data of size (num_px * num_px * 3, number of examples)

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=10000, learning_rate=0.008, print_cost=True):
    # X_train is the training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    # Y_train consists of training labels represented by a numpy array (vector) of shape (1, m_train)
    # X_test is the test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    # Y_test consists of test labels represented by a numpy array (vector) of shape (1, m_test)

    w, b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = params["w"]
    b = params["b"]
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    np.savez("model_params.npz", w = w, b = b)
    print(w, b)
    return 0


def image_test(my_image):
    num_px = 64
    params = np.load("model_params.npz")
    fname = "images/" + my_image
    image = np.array(Image.open(fname).resize((num_px, num_px)))
    image = image / 255.
    image = image.reshape((1, num_px * num_px * 3)).T
    predicted_image = predict(params["w"], params["b"], image)
    if np.squeeze(predicted_image):
        return "It's a cat"
    else:
        return "It's not a cat"


def select_file(folder_path):
    # shows the files under the given folder path
    # and makes the user select a file

    try:
        # gets the list of files
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        if not files:
            print("The provided folder path: ", folder_path)
            print("No files found in the folder.")
            return None

        # displays the files
        print("Files in the folder:\n")
        for i, file in enumerate(files):
            print(f"{i + 1}. {file}")

        # selects the file
        while True:
            choice = input("\nSelect the file: ")
            if choice.isdigit() and 1 <= int(choice) <= len(files):
                selected_file = files[int(choice) - 1]
                print(f"\nYou selected: {selected_file}")
                return selected_file
            else:
                print("Invalid choice!")

    except FileNotFoundError:
        print("Folder does not exist.")
        return None


def train_n_test():
    # initial menu
    print("Select an option(1-2)")
    print("1. Train on the given data")
    print("2. Test your image")
    n = int(input())

    if n == 1:
        # loading training data
        with h5py.File('train_catvnoncat.h5', 'r') as tr:
            train_set_x_orig = np.array(tr["train_set_x"][:])
            train_set_y = np.array(tr["train_set_y"][:])

        # loading testing data
        with h5py.File('test_catvnoncat.h5', 'r') as ts:
            test_set_x_orig = np.array(ts["test_set_x"][:])
            test_set_y = np.array(ts["test_set_y"][:])

        # flattening input data
        train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T / 255
        test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T / 255

        model(train_set_x, train_set_y, test_set_x, test_set_y)
        print("Weight and Bias for the data has been calculated and stored in model_params.npz")
        return train_n_test()
    elif n == 2:
        print(image_test(select_file("images/")))
        return 0
    else:
        print("Wrong Input!")
        return train_n_test()


if __name__ == "__main__":
    train_n_test()
