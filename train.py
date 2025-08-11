import numpy as np
import h5py

"""
# loading train data
with h5py.File('data/train.h5', 'r') as tr:
        train_set_x_orig = np.array(tr["train_set_x"][:])
        train_set_y = np.array(tr["train_set_y"][:]).reshape(1, -1)
"""

with h5py.File("data/train_cats_only.h5", "r") as tr:
    train_set_x_orig = np.array(tr["train_set_x"][:])
    train_set_y_orig = np.array(tr["train_set_y"][:])

# loading test data
with h5py.File('data/test.h5', 'r') as ts:
    test_set_x_orig = np.array(ts["test_set_x"][:])
    test_set_y = np.array(ts["test_set_y"][:]).reshape(1, -1)

train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T  # (12288, m)
train_set_y = train_set_y_orig.reshape(1, -1)  # (1, m)

# flattening input data
# train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T / 255
test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T / 255


def sigmoid(z):
    # z is a scalar or numpy array of any size.

    s = 1 / (1 + np.exp(-z))
    return s


def initialize_params(n_x, n_h, n_y):
    # n_x is the size of the input layer
    # n_h is the size of the hidden layer
    # n_y is the size of the output layer

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y,1))

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
    return parameters


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
    
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    
    return A2, cache


def compute_cost(A2, Y):
    # A2 is the output of the second activation function, of shape (n_y, m)
    # Y is the "true" labels vector of shape (n_y, m)

    m = Y.shape[1] 

    logprobs = np.multiply(Y, np.log(A2)) + np.multiply(1-Y, np.log(1-A2))
    cost = - np.sum(logprobs) / m
    

    cost = float(np.squeeze(cost))  # convert cost to a scalar 
    return cost


def backward_propagation(parameters, cache, X, Y):
    # parameters is a dictionary containing "W1", "b1", "W2", "b2"
    # cache is a dictionary containing "Z1", "A1", "Z2", "A2"
    # X is input data of shape (n_x, m)
    # Y is "true" labels vector of shape (n_y, m)

    m = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2,A1.T)/m
    db2 = np.sum(dZ2, axis = 1, keepdims = True)/m
    dZ1 = np.multiply(np.dot(W2.T,dZ2), 1 - np.power(A1,2))
    dW1 = np.dot(dZ1,X.T)/m
    db1 = np.sum(dZ1, axis = 1, keepdims = True)/m
    
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    
    return grads


def update_parameters(parameters, grads, learning_rate = 1.2):
    # parameters -- python dictionary containing your parameters 
    # grads -- python dictionary containing your gradients 

    # old parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # gradients
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    # new parameters
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
    return parameters


def nn_model(X, Y, n_h, num_iterations = 10000, learning_rate = 1.2, print_cost = True):
    # X is dataset of shape (n_x, m)
    # Y contains labels of shape (n_y, m)
    # n_h is size of the hidden layer
    # num_iterations is the number of iterations in gradient descent loop
    # print_cost, if True, prints the cost every 1000 iterations

    n_x = X.shape[0]  # size of input layer
    n_y = Y.shape[0]  # size of output layer

    parameters = initialize_params(n_x, n_h, n_y)

    # Loop: forward propagation -> cost computation -> backward propagation -> update parameter
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate)

        # prints the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    # Save the parameters in a .npz file
    np.savez("model_params.npz", W1 = parameters["W1"], b1 = parameters["b1"], W2 = parameters["W2"], b2 = parameters["b2"])
    print("Weight and Bias for the data has been calculated and stored in model_params.npz")
    return parameters

def predict(parameters, X):
    # Using the learned parameters W1, b1, W2, b2 to predict the labels for a given dataset X
    # parameters is a dictionary containing the learned parameters 
    # X is the input data of size (n_x, m)

    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    
    return predictions

nn_model(train_set_x, train_set_y, n_h = 5, num_iterations = 10000, learning_rate = 0.012, print_cost=True)

cache = np.load("model_params.npz")
parameters = {"W1": cache["W1"], "b1": cache["b1"], "W2": cache["W2"], "b2": cache["b2"]}

predictions = predict(parameters, test_set_x)
accuracy = np.mean(predictions == test_set_y) * 100
print(f"Accuracy: {float(accuracy):.2f}%")