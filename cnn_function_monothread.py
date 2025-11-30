import mnist
import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax

mnist.datasets_url = 'https://huggingface.co/spaces/chrisjay/mnist-adversarial/resolve/603879aac618aca69749a8a9172daec23a9dd2c4/files/MNIST/raw/'
mnist.temporary_dir = lambda: './minst_data'

def cnn_function_monothread(epochs=3):
    """
    Train and test a simple CNN on MNIST using a single thread.

    Parameters:
    -----------
    epochs : int
        Number of training epochs.

    Returns:
    --------
    test_loss : float
        Average loss on the test set.
    test_accuracy : float
        Accuracy on the test set.
    """

    # Load data (first 1000 images for speed)
    train_images = mnist.train_images()[:1000]
    train_labels = mnist.train_labels()[:1000]
    test_images = mnist.test_images()[:1000]
    test_labels = mnist.test_labels()[:1000]

    # Initialize layers
    conv = Conv3x3(8)                  # 28x28x1 -> 26x26x8
    pool = MaxPool2()                  # 26x26x8 -> 13x13x8
    softmax = Softmax(13 * 13 * 8, 10) # 13x13x8 -> 10

    def forward(image, label):
        out = conv.forward((image / 255) - 0.5)
        out = pool.forward(out)
        out = softmax.forward(out)

        loss = -np.log(out[label])
        acc = 1 if np.argmax(out) == label else 0

        return out, loss, acc

    def train_step(im, label, lr=0.005):
        out, loss, acc = forward(im, label)

        gradient = np.zeros(10)
        gradient[label] = -1 / out[label]

        gradient = softmax.backprop(gradient, lr)
        gradient = pool.backprop(gradient)
        gradient = conv.backprop(gradient, lr)

        return loss, acc

    print('MNIST CNN initialized!')

    # Training loop
    for epoch in range(epochs):
        print(f'--- Epoch {epoch + 1} ---')

        permutation = np.random.permutation(len(train_images))
        train_images = train_images[permutation]
        train_labels = train_labels[permutation]

        loss = 0
        num_correct = 0
        for i, (im, label) in enumerate(zip(train_images, train_labels)):
            l, acc = train_step(im, label)
            loss += l
            num_correct += acc

            if i % 100 == 99:
                print(f'[Step {i + 1}] Past 100 steps: Average Loss {loss/100:.3f} | Accuracy: {num_correct}%')
                loss = 0
                num_correct = 0

    # Testing
    print('\n--- Testing the CNN ---')
    loss = 0
    num_correct = 0
    for im, label in zip(test_images, test_labels):
        _, l, acc = forward(im, label)
        loss += l
        num_correct += acc

    num_tests = len(test_images)
    test_loss = loss / num_tests
    test_accuracy = num_correct / num_tests

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_accuracy)

    return test_loss, test_accuracy


