import numpy as np


def main():
    # Load the dataset
    mnist = np.load('datasets/sequential_mnist/mnist_dataset/mnist.npz')
    training_x = mnist['x_train']
    training_y = mnist['y_train']
    test_x = mnist['x_test']
    test_y = mnist['y_test']

    training_x = training_x.reshape(training_x.shape[0], -1)
    test_x = test_x.reshape(test_x.shape[0], -1)

    # Save the dataset
    np.savez('datasets/sequential_mnist/sequential_mnist_dataset/sequential_mnist.npz',
             x_train=training_x, y_train=training_y, x_test=test_x, y_test=test_y)
