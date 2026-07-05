# CNeural

A feedforward neural network built from scratch in C++ — no libraries, no frameworks, no autograd. Just `std::vector`, the standard library, and backpropagation derived by hand.

The goal isn't to build a fast neural network. It's to understand the machine end to end: every matrix multiply, every gradient, every weight update written and verified by hand.

## What works right now

The network learns XOR — the classic non-linearly-separable problem that requires a hidden layer. A 2→3→1 network trained with hand-written backprop:

```
epoch 0      loss 0.547586
epoch 2000   loss 0.015167
epoch 9000   loss 0.000761

--- trained predictions ---
[0,0] -> 0.008  (target 0)
[0,1] -> 0.982  (target 1)
[1,0] -> 0.983  (target 1)
[1,1] -> 0.026  (target 0)
```

## Build & run

```
make run
```

or directly:

```
g++ -std=c++17 -Wall -Wextra -g main.cpp -o cneural && ./cneural
```

Requires a C++17 compiler (GCC/Clang). No external dependencies.

## Design

- **`matrix.hpp`** — the math library. A `Matrix` struct with flat row-major storage, bounds-checked access, and seven operations: add, subtract, scalar-multiply, matrix-multiply, transpose, element-wise apply, and Hadamard product. Every operation returns a new matrix (non-destructive). Fully const-correct so chained expressions with temporaries compile cleanly.
- **`network.hpp`** — `sigmoid` and its derivative, a `Layer` (weights + bias, stores its forward-pass intermediates for backprop), a `Network` (a stack of layers), and MSE loss.
- **`main.cpp`** — the training loop.

The four backpropagation equations (Nielsen's BP1–BP4) are derived on paper and implemented directly as matrix operations:

| Equation | What it computes | CNeural operation |
|----------|------------------|-------------------|
| BP1 | Output-layer error | `(a - y).hadamard(z.apply(sigmoid_derivative))` |
| BP2 | Error propagated backward | `(W_next.transpose() * delta_next).hadamard(...)` |
| BP3 | Bias gradient | `delta` |
| BP4 | Weight gradient | `delta * input.transpose()` |

## Approach

Built one verified piece at a time, with a strict rule: look up *mechanism* (how to write something), never copy *solutions* (what to write). Every function is hand-checked on small known values before moving on.

Inspired by from-scratch low-level builders like [Tsoding](https://www.youtube.com/@TsodingDaily) and [MagicalBat](https://www.youtube.com/@magicalbat), and the math from Michael Nielsen's *[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)* and 3Blue1Brown's neural network series.

Long term, CNeural is the C++ foundation for re-implementing Andrej Karpathy's *Neural Networks: Zero to Hero* — micrograd, then language models, then transformers — built from scratch in C++ rather than Python.


