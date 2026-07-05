# CNeural

A small feedforward neural network written from scratch in C++. No libraries, no frameworks, no autograd — just `std::vector` and math I derived by hand. It learns to recognize handwritten digits from the MNIST dataset at **97.55% test accuracy**.

The point of this project wasn't to build something fast or novel. It was to actually understand how a neural network works, all the way down — every matrix multiply, every gradient, every weight update written and checked by hand instead of hidden behind a library call.

## Results

Trained on 60,000 MNIST images, tested on 10,000 it never saw during training:

```
epoch: 0   loss: 4937   accuracy: 90.60%
epoch: 5   loss: 1285   accuracy: 97.75%
epoch: 10  loss: 767    accuracy: 98.66%
epoch: 14  loss: 560    accuracy: 99.00%
training took 261 seconds

TEST accuracy: 97.55%
```

Training accuracy reaches 99% and test accuracy lands at 97.55% — the gap between them is small, which means the network actually learned what digits look like rather than just memorizing the training set.

## Running it

You need the four MNIST files (`train-images`, `train-labels`, `t10k-images`, `t10k-labels`) in a `data/` folder. Then:

```
make run
```

or directly:

```
g++ -std=c++17 -Wall -Wextra -O2 main.cpp -o cneural && ./cneural
```

Only a C++17 compiler is required. Nothing else.

## How it's put together

The code is split into a few header files, each doing one job:

- **`matrix.hpp`** — the math. A `Matrix` struct storing its numbers in a flat array, with the seven operations a neural network needs: add, subtract, scalar multiply, matrix multiply, transpose, element-wise apply, and Hadamard product. Every operation returns a new matrix instead of modifying the old one, which makes the code easier to reason about.

- **`network.hpp`** — the network itself. The sigmoid activation function and its derivative, a `Layer` (weights, bias, and the forward pass), a `Network` that chains layers together, and the mean-squared-error loss.

- **`mnist.hpp`** — reading the data. The MNIST files are raw binary, so this parses the file headers (dealing with big-endian byte order), loads the images into matrices, one-hot encodes the labels, and picks the network's best guess with argmax.

- **`main.cpp`** — the training program. Loads the data, builds a 784 → 128 → 10 network, trains it, and reports accuracy on the test set.

- **`tests.cpp`** — small hand-checks for the matrix operations and the network, built up while writing the code.

## How the learning works

A forward pass is just `sigmoid(weights × input + bias)` repeated layer by layer until you get a prediction. Learning is backpropagation: measure how wrong the prediction is, then push that error backwards through the network to work out how each weight should change, and nudge them all a little in that direction. Do this a few hundred thousand times and the random starting weights turn into something that recognizes digits.

The four backpropagation equations (from Michael Nielsen's book) map almost directly onto the matrix operations:

| Step | What it does | The code |
|------|--------------|----------|
| BP1 | error at the output layer | `(a - y).hadamard(z.apply(sigmoid_derivative))` |
| BP2 | push that error back a layer | `(next_weights.transpose() * delta).hadamard(...)` |
| BP3 | how much to change the biases | `delta` |
| BP4 | how much to change the weights | `delta * input.transpose()` |

## Notes

I built this one piece at a time, testing each function on small examples I could check by hand before moving on. The rule throughout was to look up *how* to write something when I got stuck, but never to copy a finished solution — the whole point was to end up understanding every line.

The math comes from Michael Nielsen's [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/) and 3Blue1Brown's neural network videos. The from-scratch, build-it-in-C style was inspired by [Tsoding](https://www.youtube.com/@TsodingDaily) and [MagicalBat](https://www.youtube.com/@magicalbat).
